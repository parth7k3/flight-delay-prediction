"""
Stage 4 — Multi-Million-Row Chronological Split & Training-Fold SMOTE (balance_classes.py)
Module Owner: Krishna Sharma (S24CSEU2756)

Responsibilities:
  1. Partitions the ~5.8M+ real 2024 flight dataset chronologically (80% Jan–Oct Train,
     20% Oct–Dec Untouched Test = ~1.1M+ real test flights) with zero temporal data leakage.
  2. Draws a stratified chronological training partition (300,000 real flights) for tractable
     exact k-NN SMOTE oversampling (balancing Class 0 & Class 1 50/50 -> ~480,000 vectors)
     while also preserving a chronologically late validation slice for threshold calibration.
  3. Leaves the 1,000,000+ row Chronological Test Split 100% untouched at its true operational skew.
"""

from __future__ import annotations
import json
import time
import duckdb
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE

from config import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    MODEL_FEATURE_COLUMNS,
    SMOTE_MAX_TRAIN_SAMPLE,
    RANDOM_SEED,
)


def run_balance_classes() -> dict:
    t0 = time.perf_counter()
    feat_parquet = PROCESSED_DIR / "featured_flights.parquet"
    with open(OUTPUTS_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
        feat_meta = json.load(f)

    cutoff_ts = feat_meta["chronological_cutoff_timestamp"]
    feat_cols_sql = ", ".join(MODEL_FEATURE_COLUMNS)

    con = duckdb.connect()
    try:
        # Full multi-million-row partition statistics
        full_train_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN DEP_DEL15 = 0 THEN 1 ELSE 0 END) AS c0,
                SUM(CASE WHEN DEP_DEL15 = 1 THEN 1 ELSE 0 END) AS c1,
                MIN(CRS_DEP_TIMESTAMP) AS min_ts,
                MAX(CRS_DEP_TIMESTAMP) AS max_ts
            FROM read_parquet('{feat_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            """
        ).fetchone()

        full_test_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN DEP_DEL15 = 0 THEN 1 ELSE 0 END) AS c0,
                SUM(CASE WHEN DEP_DEL15 = 1 THEN 1 ELSE 0 END) AS c1,
                MIN(CRS_DEP_TIMESTAMP) AS min_ts,
                MAX(CRS_DEP_TIMESTAMP) AS max_ts
            FROM read_parquet('{feat_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP >= '{cutoff_ts}'
            """
        ).fetchone()

        # Sample SMOTE_MAX_TRAIN_SAMPLE real flights strictly from the 80% training partition
        train_sample_df = con.execute(
            f"""
            SELECT {feat_cols_sql}, DEP_DEL15, DEP_DELAY
            FROM read_parquet('{feat_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            USING SAMPLE reservoir({SMOTE_MAX_TRAIN_SAMPLE} ROWS) REPEATABLE ({RANDOM_SEED})
            """
        ).fetchdf()

        # Stratified validation slice across the 80% training window for unbiased threshold calibration
        cal_df = con.execute(
            f"""
            SELECT {feat_cols_sql}, DEP_DEL15
            FROM read_parquet('{feat_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP < '{cutoff_ts}'
            USING SAMPLE reservoir(50000 ROWS) REPEATABLE (99)
            """
        ).fetchdf()
    finally:
        con.close()

    X_train_raw = train_sample_df[MODEL_FEATURE_COLUMNS].astype(np.float32)
    y_train_raw = train_sample_df["DEP_DEL15"].astype(np.int32)
    y_train_reg = train_sample_df["DEP_DELAY"].astype(np.float32)

    def _snap_to_sensor_grid(df_in):
        df_out = df_in.copy()
        for int_col in [
            "DISTANCE", "CRS_ELAPSED_TIME", "is_weekend", "has_inbound_leg",
            "TEMP_F", "DEW_POINT_F", "WIND_SPEED_KT", "wind_max_3h",
            "origin_rolling_3h_vol", "prev_aircraft_delay_t2h",
        ]:
            df_out[int_col] = np.round(df_out[int_col]).astype(np.float32)
        df_out["temp_dew_spread"] = (df_out["TEMP_F"] - df_out["DEW_POINT_F"]).astype(np.float32)
        df_out["VISIBILITY_MI"] = (np.round(df_out["VISIBILITY_MI"] * 2.0) / 2.0).astype(np.float32)
        df_out["vis_min_3h"] = (np.round(df_out["vis_min_3h"] * 2.0) / 2.0).astype(np.float32)
        df_out["PRESSURE_MB"] = np.round(df_out["PRESSURE_MB"], 0).astype(np.float32)
        df_out["PRECIP_IN"] = np.round(df_out["PRECIP_IN"], 2).astype(np.float32)
        df_out["precip_sum_3h"] = np.round(df_out["precip_sum_3h"], 2).astype(np.float32)
        return df_out

    X_train_raw = _snap_to_sensor_grid(X_train_raw)
    smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=RANDOM_SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
    X_train_smote = _snap_to_sensor_grid(X_train_smote)

    # Project interpolated (dep_hour_sin, dep_hour_cos) back onto unit circle r=1 (hourly 15-min slots)
    hr_ang = np.mod(np.arctan2(X_train_smote["dep_hour_sin"], X_train_smote["dep_hour_cos"]), 2.0 * np.pi)
    slot_idx = np.round(hr_ang / (2.0 * np.pi / 96.0)) % 96
    exact_hr_ang = 2.0 * np.pi * slot_idx / 96.0
    X_train_smote["dep_hour_sin"] = np.sin(exact_hr_ang).astype(np.float32)
    X_train_smote["dep_hour_cos"] = np.cos(exact_hr_ang).astype(np.float32)

    # Snap interpolated (dow_sin, dow_cos) to nearest of the 7 discrete days of week
    dow_ang = np.mod(np.arctan2(X_train_smote["dow_sin"], X_train_smote["dow_cos"]), 2.0 * np.pi)
    dow_idx = np.round(dow_ang / (2.0 * np.pi / 7.0)) % 7
    exact_dow_ang = 2.0 * np.pi * dow_idx / 7.0
    X_train_smote["dow_sin"] = np.sin(exact_dow_ang).astype(np.float32)
    X_train_smote["dow_cos"] = np.cos(exact_dow_ang).astype(np.float32)

    cal_features = _snap_to_sensor_grid(cal_df[MODEL_FEATURE_COLUMNS].astype(np.float32))

    splits_bundle = {
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "y_train_reg": y_train_reg,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_cal": cal_features,
        "y_cal": cal_df["DEP_DEL15"].astype(np.int32),
        "cutoff_ts": cutoff_ts,
    }
    joblib.dump(splits_bundle, PROCESSED_DIR / "train_test_splits.joblib")

    full_tr_total, full_tr_c0, full_tr_c1, tr_start, tr_end = full_train_stats
    full_te_total, full_te_c0, full_te_c1, te_start, te_end = full_test_stats

    summary = {
        "chronological_split_ratio": "80/20",
        "cutoff_timestamp": cutoff_ts,
        "train_period_start": str(tr_start),
        "train_period_end": str(tr_end),
        "test_period_start": str(te_start),
        "test_period_end": str(te_end),
        "full_train_partition": {
            "total": int(full_tr_total),
            "on_time_class_0": int(full_tr_c0),
            "delayed_class_1": int(full_tr_c1),
            "delay_pct": round(float(full_tr_c1 / max(1, full_tr_total) * 100.0), 2),
        },
        "train_before_smote": {
            "total": int(len(y_train_raw)),
            "on_time_class_0": int((y_train_raw == 0).sum()),
            "delayed_class_1": int((y_train_raw == 1).sum()),
            "delay_pct": round(float(y_train_raw.mean() * 100.0), 2),
        },
        "train_after_smote": {
            "total": int(len(y_train_smote)),
            "on_time_class_0": int((y_train_smote == 0).sum()),
            "delayed_class_1": int((y_train_smote == 1).sum()),
            "synthetic_samples_added": int(len(y_train_smote) - len(y_train_raw)),
            "delay_pct": round(float(y_train_smote.mean() * 100.0), 2),
        },
        "test_untouched": {
            "total": int(full_te_total),
            "on_time_class_0": int(full_te_c0),
            "delayed_class_1": int(full_te_c1),
            "delay_pct": round(float(full_te_c1 / max(1, full_te_total) * 100.0), 2),
        },
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    with open(OUTPUTS_DIR / "smote_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[balance_classes.py] Full Chronological Train: {int(full_tr_total):,} | "
        f"SMOTE Train Sample: {len(y_train_raw):,} -> {len(y_train_smote):,} (50/50 balanced) | "
        f"Untouched Chronological Test: {int(full_te_total):,} ({summary['test_untouched']['delay_pct']}% delayed) [{summary['elapsed_sec']}s]."
    )
    return summary


if __name__ == "__main__":
    run_balance_classes()
