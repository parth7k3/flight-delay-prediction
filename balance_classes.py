"""
Stage 4 — Chronological Train/Test Partitioning & Training-Only SMOTE Module (balance_classes.py)
Module Owner: Krishna Sharma (S24CSEU2756)

Responsibilities:
  1. Splits dataset chronologically (80% earliest flights -> Train, 20% latest flights -> Test)
     to prevent temporal data leakage (future weather/schedule waves never leak into training).
  2. Applies SMOTE synthetic oversampling strictly to the 80% training partition so Class 0 and Class 1
     are balanced 50/50 during classifier updates.
  3. Preserves the 20% chronological test partition completely untouched with true ~80/20 operational skew.
"""

from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from config import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    CHRONOLOGICAL_TRAIN_RATIO,
    MODEL_FEATURE_COLUMNS,
    RANDOM_SEED,
)


def run_balance_classes() -> dict:
    t0 = time.perf_counter()
    df = pd.read_parquet(PROCESSED_DIR / "featured_flights.parquet")
    df = df.sort_values("CRS_DEP_TIMESTAMP", ascending=True).reset_index(drop=True)

    split_idx = int(len(df) * CHRONOLOGICAL_TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train_raw = train_df[MODEL_FEATURE_COLUMNS].astype(np.float32)
    y_train_raw = train_df["DEP_DEL15"].astype(int)
    y_train_reg = train_df["DEP_DELAY"].astype(np.float32)

    X_test = test_df[MODEL_FEATURE_COLUMNS].astype(np.float32)
    y_test = test_df["DEP_DEL15"].astype(int)
    y_test_reg = test_df["DEP_DELAY"].astype(np.float32)

    # Apply SMOTE strictly on the training partition (Belcastro et al., 2016)
    smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=RANDOM_SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)

    # Quantize discrete count/binary features post-SMOTE so tree ensembles do not split on synthetic fractional artifacts
    X_train_smote["is_weekend"] = np.round(X_train_smote["is_weekend"])
    X_train_smote["origin_rolling_3h_vol"] = np.round(X_train_smote["origin_rolling_3h_vol"])
    X_train_smote["DISTANCE"] = np.round(X_train_smote["DISTANCE"])

    splits_bundle = {
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "y_train_reg": y_train_reg,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_reg": y_test_reg,
        "train_dates": (str(train_df["CRS_DEP_TIMESTAMP"].iloc[0]), str(train_df["CRS_DEP_TIMESTAMP"].iloc[-1])),
        "test_dates": (str(test_df["CRS_DEP_TIMESTAMP"].iloc[0]), str(test_df["CRS_DEP_TIMESTAMP"].iloc[-1])),
        "test_meta": test_df[[
            "FL_DATE", "CRS_DEP_TIMESTAMP", "OP_UNIQUE_CARRIER", "AIRLINE_NAME",
            "ORIGIN", "DEST", "DISTANCE", "WIND_SPEED_KT", "VISIBILITY_MI",
            "PRECIP_IN", "TEMP_F", "DEP_DELAY", "DEP_DEL15"
        ]].reset_index(drop=True),
    }
    joblib.dump(splits_bundle, PROCESSED_DIR / "train_test_splits.joblib")

    summary = {
        "chronological_split_ratio": "80/20",
        "train_period_start": splits_bundle["train_dates"][0],
        "train_period_end": splits_bundle["train_dates"][1],
        "test_period_start": splits_bundle["test_dates"][0],
        "test_period_end": splits_bundle["test_dates"][1],
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
            "total": int(len(y_test)),
            "on_time_class_0": int((y_test == 0).sum()),
            "delayed_class_1": int((y_test == 1).sum()),
            "delay_pct": round(float(y_test.mean() * 100.0), 2),
        },
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    with open(OUTPUTS_DIR / "smote_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[balance_classes.py] Train raw: {len(y_train_raw):,} ({summary['train_before_smote']['delay_pct']}% delayed) "
        f"-> Train SMOTE: {len(y_train_smote):,} (50.0% balanced) | "
        f"Untouched Test: {len(y_test):,} ({summary['test_untouched']['delay_pct']}% delayed)."
    )
    return summary


if __name__ == "__main__":
    run_balance_classes()
