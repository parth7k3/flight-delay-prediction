"""
Stage 3 — Cyclical Temporal Encoding & Hub Congestion Feature Engineering (feature_engineering.py)
Module Owner: Vipul Gupta (S24CSEU2526)

Responsibilities:
  1. Converts scheduled departure times into 24-hour sine/cosine circular encodings so 23:55 and 00:05
     remain adjacent in Euclidean space:
       dep_hour_sin = sin(2 * pi * hour / 24),  dep_hour_cos = cos(2 * pi * hour / 24)
  2. Encodes day-of-week cyclically (dow_sin, dow_cos) and weekend indicator.
  3. Computes rolling 3-hour origin airport departure volume (origin_rolling_3h_vol) and local runway
     congestion ratio (origin_congestion_ratio) to capture downstream ripple effects.
  4. Computes leak-free historical carrier/origin/destination delay baselines strictly on the
     chronological training window (first 80%) and maps them to all records.
"""

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    CHRONOLOGICAL_TRAIN_RATIO,
    MODEL_FEATURE_COLUMNS,
)


def encode_cyclical_time(dep_hour_float: np.ndarray | float, day_of_week: np.ndarray | int) -> dict:
    angle_hr = 2.0 * np.pi * (dep_hour_float / 24.0)
    angle_dow = 2.0 * np.pi * (day_of_week / 7.0)
    return {
        "dep_hour_sin": np.sin(angle_hr),
        "dep_hour_cos": np.cos(angle_hr),
        "dow_sin": np.sin(angle_dow),
        "dow_cos": np.cos(angle_dow),
    }


def run_feature_engineering() -> dict:
    t0 = time.perf_counter()
    in_parquet = PROCESSED_DIR / "cleaned_joined_flights.parquet"
    df = pd.read_parquet(in_parquet)

    # Ensure strict chronological ordering before rolling aggregations & historical baselines
    df = df.sort_values("CRS_DEP_TIMESTAMP", ascending=True).reset_index(drop=True)
    dep_dt = pd.to_datetime(df["CRS_DEP_TIMESTAMP"])

    dep_hour_float = dep_dt.dt.hour + dep_dt.dt.minute / 60.0
    dow = dep_dt.dt.dayofweek
    cyc = encode_cyclical_time(dep_hour_float.to_numpy(), dow.to_numpy())

    df["dep_hour_float"] = np.round(dep_hour_float, 3)
    df["dep_hour_sin"] = cyc["dep_hour_sin"]
    df["dep_hour_cos"] = cyc["dep_hour_cos"]
    df["dow_sin"] = cyc["dow_sin"]
    df["dow_cos"] = cyc["dow_cos"]
    df["is_weekend"] = dow.isin([5, 6]).astype(int)

    # Meteorological dew-point spread (fog / saturation proximity)
    df["temp_dew_spread"] = np.round(df["TEMP_F"] - df["DEW_POINT_F"], 2)

    # Rolling 3-hour origin airport departure volume (local runway load)
    df["dep_hour_bucket"] = dep_dt.dt.floor("h")
    hourly_vol = (
        df.groupby(["ORIGIN", "dep_hour_bucket"])
        .size()
        .rename("origin_1h_vol")
        .reset_index()
        .sort_values(["ORIGIN", "dep_hour_bucket"])
    )
    hourly_vol["origin_rolling_3h_vol"] = (
        hourly_vol.groupby("ORIGIN")["origin_1h_vol"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )

    df = df.merge(
        hourly_vol[["ORIGIN", "dep_hour_bucket", "origin_rolling_3h_vol"]],
        on=["ORIGIN", "dep_hour_bucket"],
        how="left",
    )
    expected_3h_cap = np.maximum(df["ORIGIN_CAPACITY_HR"] * 0.42, 1.0)
    df["origin_congestion_ratio"] = np.round(df["origin_rolling_3h_vol"] / expected_3h_cap, 3)

    # Re-sort chronologically after merge
    df = df.sort_values("CRS_DEP_TIMESTAMP", ascending=True).reset_index(drop=True)

    # Compute historical carrier/origin/dest delay priors STRICTLY on the first 80% chronological window
    split_idx = int(len(df) * CHRONOLOGICAL_TRAIN_RATIO)
    train_slice = df.iloc[:split_idx]
    global_prior = float(train_slice["DEP_DEL15"].mean())

    carrier_rates = train_slice.groupby("OP_UNIQUE_CARRIER")["DEP_DEL15"].mean().to_dict()
    origin_rates = train_slice.groupby("ORIGIN")["DEP_DEL15"].mean().to_dict()
    dest_rates = train_slice.groupby("DEST")["DEP_DEL15"].mean().to_dict()

    # Also compute typical hourly rolling volume lookup per (ORIGIN, hour_of_day) for live Streamlit queries
    df["_hour_int"] = dep_dt.dt.hour
    typical_hub_vol = (
        train_slice.assign(_hour_int=pd.to_datetime(train_slice["CRS_DEP_TIMESTAMP"]).dt.hour)
        .groupby(["ORIGIN", "_hour_int"])["origin_rolling_3h_vol"]
        .median()
        .reset_index()
    )
    hub_vol_lookup = {
        f"{row['ORIGIN']}_{int(row['_hour_int'])}": float(row["origin_rolling_3h_vol"])
        for _, row in typical_hub_vol.iterrows()
    }

    df["carrier_hist_delay_rate"] = df["OP_UNIQUE_CARRIER"].map(carrier_rates).fillna(global_prior)
    df["origin_hist_delay_rate"] = df["ORIGIN"].map(origin_rates).fillna(global_prior)
    df["dest_hist_delay_rate"] = df["DEST"].map(dest_rates).fillna(global_prior)
    df = df.drop(columns=["dep_hour_bucket", "_hour_int"])

    out_parquet = PROCESSED_DIR / "featured_flights.parquet"
    df.to_parquet(out_parquet, index=False)

    lookup_artifacts = {
        "global_delay_prior": round(global_prior, 4),
        "carrier_hist_delay_rate": {k: round(float(v), 4) for k, v in carrier_rates.items()},
        "origin_hist_delay_rate": {k: round(float(v), 4) for k, v in origin_rates.items()},
        "dest_hist_delay_rate": {k: round(float(v), 4) for k, v in dest_rates.items()},
        "hub_hourly_vol_lookup": hub_vol_lookup,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "total_featured_rows": int(len(df)),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    with open(OUTPUTS_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lookup_artifacts, f, indent=2)

    print(
        f"[feature_engineering.py] Engineered {len(MODEL_FEATURE_COLUMNS)} features across "
        f"{len(df):,} chronological flights in {lookup_artifacts['elapsed_sec']}s."
    )
    return lookup_artifacts


if __name__ == "__main__":
    run_feature_engineering()
