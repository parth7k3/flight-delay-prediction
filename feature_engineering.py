"""
Stage 3 — Out-of-Core Cyclical Time & Hub Congestion Feature Engineering (feature_engineering.py)
Module Owner: Vipul Gupta (S24CSEU2526)

Responsibilities:
  1. Computes 24-hour circular departure encodings (dep_hour_sin, dep_hour_cos), day-of-week
     cyclical encodings (dow_sin, dow_cos), and seasonal month cyclical encodings (month_sin, month_cos)
     across 5M–6M+ joined real flights out-of-core via DuckDB SQL.
  2. Computes exact rolling 3-hour origin hub departure volume (origin_rolling_3h_vol) and
     runway congestion ratio (origin_congestion_ratio) using SQL window functions.
  3. Computes leak-free historical carrier/origin/destination delay baselines strictly on the
     chronological first 80% training window.
  4. Exports full multi-million-row featured Parquet + a compact 60,000-row stratified empirical
     sample (dashboard_sample.parquet) for instant UI charts & cloud portability.
"""

from __future__ import annotations
import json
import time
import duckdb
import numpy as np

from config import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    CHRONOLOGICAL_TRAIN_RATIO,
    MODEL_FEATURE_COLUMNS,
)


def encode_cyclical_time(
    dep_hour_float: np.ndarray | float,
    day_of_week: np.ndarray | int,
    month_num: np.ndarray | int = 6,
) -> dict:
    angle_hr = 2.0 * np.pi * (dep_hour_float / 24.0)
    angle_dow = 2.0 * np.pi * (day_of_week / 7.0)
    angle_mo = 2.0 * np.pi * (month_num / 12.0)
    return {
        "dep_hour_sin": np.sin(angle_hr),
        "dep_hour_cos": np.cos(angle_hr),
        "dow_sin": np.sin(angle_dow),
        "dow_cos": np.cos(angle_dow),
        "month_sin": np.sin(angle_mo),
        "month_cos": np.cos(angle_mo),
    }


def run_feature_engineering() -> dict:
    t0 = time.perf_counter()
    in_parquet = PROCESSED_DIR / "cleaned_joined_flights.parquet"
    out_parquet = PROCESSED_DIR / "featured_flights.parquet"
    sample_parquet = PROCESSED_DIR / "dashboard_sample.parquet"
    out_parquet.unlink(missing_ok=True)
    sample_parquet.unlink(missing_ok=True)

    con = duckdb.connect()
    try:
        total_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{in_parquet.as_posix()}')").fetchone()[0])
        split_offset = int(total_rows * CHRONOLOGICAL_TRAIN_RATIO)

        # Exact chronological 80% split timestamp (zero future temporal leakage)
        cutoff_ts = str(
            con.execute(
                f"""
                SELECT CRS_DEP_TIMESTAMP
                FROM read_parquet('{in_parquet.as_posix()}')
                ORDER BY CRS_DEP_TIMESTAMP ASC
                LIMIT 1 OFFSET {split_offset}
                """
            ).fetchone()[0]
        )

        # Compute leak-free historical delay priors strictly on chronological training partition
        global_prior = float(
            con.execute(
                f"SELECT AVG(CAST(DEP_DEL15 AS DOUBLE)) FROM read_parquet('{in_parquet.as_posix()}') WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'"
            ).fetchone()[0]
        )

        carrier_df = con.execute(
            f"""
            SELECT OP_UNIQUE_CARRIER, AVG(CAST(DEP_DEL15 AS DOUBLE)) AS rate
            FROM read_parquet('{in_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            GROUP BY 1
            """
        ).fetchdf()
        origin_df = con.execute(
            f"""
            SELECT ORIGIN, AVG(CAST(DEP_DEL15 AS DOUBLE)) AS rate
            FROM read_parquet('{in_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            GROUP BY 1
            """
        ).fetchdf()
        dest_df = con.execute(
            f"""
            SELECT DEST, (SUM(CAST(DEP_DEL15 AS DOUBLE)) + 20.0 * {global_prior}) / (COUNT(*) + 20.0) AS rate
            FROM read_parquet('{in_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            GROUP BY 1
            """
        ).fetchdf()

        con.register("carrier_priors_v", carrier_df)
        con.register("origin_priors_v", origin_df)
        con.register("dest_priors_v", dest_df)

        # Out-of-core feature engineering + rolling 3h hub load + zero-leakage T-2h tail & hub cascade telemetry
        con.execute(
            f"""
            COPY (
                WITH hourly_hub_load AS (
                    SELECT
                        ORIGIN,
                        DEP_HOUR_TS,
                        COUNT(*) AS hr_deps,
                        SUM(CAST(DEP_DEL15 AS DOUBLE)) AS hr_del15
                    FROM read_parquet('{in_parquet.as_posix()}')
                    GROUP BY 1, 2
                ),
                rolling_hub_load AS (
                    SELECT
                        ORIGIN,
                        DEP_HOUR_TS,
                        CAST(
                            SUM(hr_deps) OVER (
                                PARTITION BY ORIGIN
                                ORDER BY DEP_HOUR_TS
                                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                            ) AS FLOAT
                        ) AS origin_rolling_3h_vol,
                        CAST(
                            COALESCE(
                                SUM(hr_del15) OVER (
                                    PARTITION BY ORIGIN
                                    ORDER BY DEP_HOUR_TS
                                    ROWS BETWEEN 5 PRECEDING AND 2 PRECEDING
                                ) / NULLIF(
                                    SUM(hr_deps) OVER (
                                        PARTITION BY ORIGIN
                                        ORDER BY DEP_HOUR_TS
                                        ROWS BETWEEN 5 PRECEDING AND 2 PRECEDING
                                    ), 0
                                ),
                                {global_prior}
                            ) AS FLOAT
                        ) AS hub_t2h_delay_rate
                    FROM hourly_hub_load
                ),
                tail_lags AS (
                    SELECT
                        *,
                        CASE WHEN TAIL_NUM IS NOT NULL AND TAIL_NUM != '' THEN
                            LAG(CAST(CRS_DEP_TIMESTAMP AS TIMESTAMP), 1) OVER (PARTITION BY TAIL_NUM, FL_DATE ORDER BY CRS_DEP_TIMESTAMP)
                        ELSE NULL END AS lag1_ts,
                        CASE WHEN TAIL_NUM IS NOT NULL AND TAIL_NUM != '' THEN
                            LAG(DEP_DELAY, 1) OVER (PARTITION BY TAIL_NUM, FL_DATE ORDER BY CRS_DEP_TIMESTAMP)
                        ELSE NULL END AS lag1_delay,
                        CASE WHEN TAIL_NUM IS NOT NULL AND TAIL_NUM != '' THEN
                            LAG(DEP_DELAY, 2) OVER (PARTITION BY TAIL_NUM, FL_DATE ORDER BY CRS_DEP_TIMESTAMP)
                        ELSE NULL END AS lag2_delay
                    FROM read_parquet('{in_parquet.as_posix()}')
                ),
                enriched AS (
                    SELECT
                        f.FL_DATE,
                        f.CRS_DEP_TIMESTAMP,
                        f.OP_UNIQUE_CARRIER,
                        f.AIRLINE_NAME,
                        f.ORIGIN,
                        f.DEST,
                        f.WEATHER_CONDITION,
                        CAST(f.DEP_DELAY AS FLOAT) AS DEP_DELAY,
                        CAST(f.DEP_DEL15 AS INTEGER) AS DEP_DEL15,
                        CAST(f.DISTANCE AS FLOAT) AS DISTANCE,
                        CAST(f.CRS_ELAPSED_TIME AS FLOAT) AS CRS_ELAPSED_TIME,
                        CAST((CAST(f.CRS_DEP_TIME / 100 AS INTEGER)) + (f.CRS_DEP_TIME % 100) / 60.0 AS FLOAT) AS dep_hour_float,
                        CAST(DAYOFWEEK(CAST(f.FL_DATE AS DATE)) AS INTEGER) AS dow_num,
                        CAST(MONTH(CAST(f.FL_DATE AS DATE)) AS INTEGER) AS month_num,
                        CAST(f.TEMP_F AS FLOAT) AS TEMP_F,
                        CAST(f.DEW_POINT_F AS FLOAT) AS DEW_POINT_F,
                        CAST(ROUND(f.TEMP_F - f.DEW_POINT_F, 2) AS FLOAT) AS temp_dew_spread,
                        CAST(f.WIND_SPEED_KT AS FLOAT) AS WIND_SPEED_KT,
                        CAST(f.VISIBILITY_MI AS FLOAT) AS VISIBILITY_MI,
                        CAST(f.PRECIP_IN AS FLOAT) AS PRECIP_IN,
                        CAST(f.PRESSURE_MB AS FLOAT) AS PRESSURE_MB,
                        CAST(f.wind_max_3h AS FLOAT) AS wind_max_3h,
                        CAST(f.precip_sum_3h AS FLOAT) AS precip_sum_3h,
                        CAST(f.vis_min_3h AS FLOAT) AS vis_min_3h,
                        r.origin_rolling_3h_vol,
                        CAST(ROUND(r.origin_rolling_3h_vol / GREATEST(f.ORIGIN_CAPACITY_HR * 1.6, 10.0), 3) AS FLOAT) AS origin_congestion_ratio,
                        CAST(
                            LEAST(300.0, GREATEST(-20.0, COALESCE(
                                CASE
                                    WHEN f.lag1_ts IS NOT NULL AND DATE_DIFF('minute', f.lag1_ts, CAST(f.CRS_DEP_TIMESTAMP AS TIMESTAMP)) >= 115 THEN f.lag1_delay
                                    WHEN f.lag2_delay IS NOT NULL THEN f.lag2_delay
                                    ELSE 0.0
                                END,
                                0.0
                            ))) AS FLOAT
                        ) AS prev_aircraft_delay_t2h,
                        CAST(CASE WHEN f.lag1_ts IS NOT NULL THEN 1.0 ELSE 0.0 END AS FLOAT) AS has_inbound_leg,
                        COALESCE(r.hub_t2h_delay_rate, CAST({global_prior} AS FLOAT)) AS origin_t2h_delay_rate,
                        COALESCE(rd.hub_t2h_delay_rate, CAST({global_prior} AS FLOAT)) AS dest_t2h_delay_rate,
                        CAST(COALESCE(cp.rate, {global_prior}) AS FLOAT) AS carrier_hist_delay_rate,
                        CAST(COALESCE(op.rate, {global_prior}) AS FLOAT) AS origin_hist_delay_rate,
                        CAST(COALESCE(dp.rate, {global_prior}) AS FLOAT) AS dest_hist_delay_rate
                    FROM tail_lags f
                    LEFT JOIN rolling_hub_load r
                        ON f.ORIGIN = r.ORIGIN AND f.DEP_HOUR_TS = r.DEP_HOUR_TS
                    LEFT JOIN rolling_hub_load rd
                        ON f.DEST = rd.ORIGIN AND f.DEP_HOUR_TS = rd.DEP_HOUR_TS
                    LEFT JOIN carrier_priors_v cp
                        ON f.OP_UNIQUE_CARRIER = cp.OP_UNIQUE_CARRIER
                    LEFT JOIN origin_priors_v op
                        ON f.ORIGIN = op.ORIGIN
                    LEFT JOIN dest_priors_v dp
                        ON f.DEST = dp.DEST
                )
                SELECT
                    FL_DATE,
                    CRS_DEP_TIMESTAMP,
                    OP_UNIQUE_CARRIER,
                    AIRLINE_NAME,
                    ORIGIN,
                    DEST,
                    WEATHER_CONDITION,
                    DEP_DELAY,
                    DEP_DEL15,
                    DISTANCE,
                    CRS_ELAPSED_TIME,
                    dep_hour_float,
                    CAST(SIN(2.0 * PI() * dep_hour_float / 24.0) AS FLOAT) AS dep_hour_sin,
                    CAST(COS(2.0 * PI() * dep_hour_float / 24.0) AS FLOAT) AS dep_hour_cos,
                    CAST(SIN(2.0 * PI() * dow_num / 7.0) AS FLOAT) AS dow_sin,
                    CAST(COS(2.0 * PI() * dow_num / 7.0) AS FLOAT) AS dow_cos,
                    CAST(SIN(2.0 * PI() * month_num / 12.0) AS FLOAT) AS month_sin,
                    CAST(COS(2.0 * PI() * month_num / 12.0) AS FLOAT) AS month_cos,
                    CAST(CASE WHEN dow_num IN (0, 6) THEN 1.0 ELSE 0.0 END AS FLOAT) AS is_weekend,
                    TEMP_F,
                    DEW_POINT_F,
                    temp_dew_spread,
                    WIND_SPEED_KT,
                    VISIBILITY_MI,
                    PRECIP_IN,
                    PRESSURE_MB,
                    wind_max_3h,
                    precip_sum_3h,
                    vis_min_3h,
                    origin_rolling_3h_vol,
                    origin_congestion_ratio,
                    prev_aircraft_delay_t2h,
                    has_inbound_leg,
                    origin_t2h_delay_rate,
                    dest_t2h_delay_rate,
                    carrier_hist_delay_rate,
                    origin_hist_delay_rate,
                    dest_hist_delay_rate
                FROM enriched
                ORDER BY CRS_DEP_TIMESTAMP ASC
            ) TO '{out_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        # Typical hourly rolling hub volume lookup for live single-flight queries in Streamlit
        vol_df = con.execute(
            f"""
            SELECT
                ORIGIN,
                CAST(FLOOR(dep_hour_float) AS INTEGER) AS hr_int,
                MEDIAN(origin_rolling_3h_vol) AS med_vol
            FROM read_parquet('{out_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            GROUP BY 1, 2
            """
        ).fetchdf()

        # Export compact 60,000-row representative empirical sample for UI charts & GitHub portability
        con.execute(
            f"""
            COPY (
                SELECT
                    FL_DATE, CRS_DEP_TIMESTAMP, OP_UNIQUE_CARRIER, AIRLINE_NAME,
                    ORIGIN, DEST, WEATHER_CONDITION, dep_hour_float,
                    WIND_SPEED_KT, VISIBILITY_MI, PRECIP_IN, TEMP_F,
                    origin_congestion_ratio, DISTANCE, DEP_DELAY, DEP_DEL15
                FROM read_parquet('{out_parquet.as_posix()}')
                USING SAMPLE reservoir(60000 ROWS) REPEATABLE (42)
            ) TO '{sample_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    hub_vol_lookup = {
        f"{row['ORIGIN']}_{int(row['hr_int'])}": round(float(row["med_vol"]), 1)
        for _, row in vol_df.iterrows()
    }

    lookup_artifacts = {
        "chronological_cutoff_timestamp": cutoff_ts,
        "global_delay_prior": round(global_prior, 4),
        "carrier_hist_delay_rate": {r["OP_UNIQUE_CARRIER"]: round(float(r["rate"]), 4) for _, r in carrier_df.iterrows()},
        "origin_hist_delay_rate": {r["ORIGIN"]: round(float(r["rate"]), 4) for _, r in origin_df.iterrows()},
        "dest_hist_delay_rate": {r["DEST"]: round(float(r["rate"]), 4) for _, r in dest_df.iterrows()},
        "hub_hourly_vol_lookup": hub_vol_lookup,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "total_featured_rows": total_rows,
        "featured_parquet_mb": round(out_parquet.stat().st_size / (1024 * 1024), 1),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    with open(OUTPUTS_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lookup_artifacts, f, indent=2)

    print(
        f"[feature_engineering.py] Engineered {len(MODEL_FEATURE_COLUMNS)} features out-of-core across "
        f"{total_rows:,} real flights (80% cutoff: {cutoff_ts}) in {lookup_artifacts['elapsed_sec']}s."
    )
    return lookup_artifacts


if __name__ == "__main__":
    run_feature_engineering()
