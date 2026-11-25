"""
Stage 5 — Classifier & Regression Model Training with Cross-Validation (train_models.py)
Module Owner: Krishna Sharma (S24CSEU2756)

Responsibilities:
  1. Runs stratified/time-aware cross-validation on the training fold (with fold-internal SMOTE
     verification to guard against optimistic leakage).
  2. Trains the candidate classification architectures on the SMOTE-balanced training set:
       - Logistic Regression (Baseline) [with StandardScaler]
       - Random Forest (100 Trees)
       - Gradient Boosting Classifier
       - XGBoost Classifier (Tuned)
  3. Trains the Secondary Continuous Regression Baseline (XGBoost Regressor) on raw training delays
     to estimate expected departure delay length in minutes (Chapter 1 Problem Formulation).
  4. Packages all trained models, scalers, and lookup metadata into models/flight_delay_models.joblib.
"""

from __future__ import annotations
import json
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor

from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    MODEL_FEATURE_COLUMNS,
    RANDOM_SEED,
)


def run_train_models() -> dict:
    t0 = time.perf_counter()
    splits = joblib.load(PROCESSED_DIR / "train_test_splits.joblib")
    with open(OUTPUTS_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
        feature_meta = json.load(f)

    X_train_raw = splits["X_train_raw"]
    y_train_raw = splits["y_train_raw"]
    y_train_reg = splits["y_train_reg"]
    X_train_smote = splits["X_train_smote"]
    y_train_smote = splits["y_train_smote"]

    # Define Primary Classification Models
    classifiers = {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest (100 Trees)": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=12,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=95,
            learning_rate=0.09,
            max_depth=4,
            subsample=0.85,
            random_state=RANDOM_SEED,
        ),
        "XGBoost Classifier (Tuned)": XGBClassifier(
            n_estimators=240,
            learning_rate=0.06,
            max_depth=6,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.3,
            reg_lambda=2.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }

    # 3-Fold Fold-Internal SMOTE Cross-Validation on a 12,000-row chronological training subset
    cv_subset_n = min(12000, len(X_train_raw))
    X_cv = X_train_raw.iloc[-cv_subset_n:]
    y_cv = y_train_raw.iloc[-cv_subset_n:]
    skf = StratifiedKFold(n_splits=3, shuffle=False)

    cv_scores = {}
    trained_classifiers = {}

    for name, model in classifiers.items():
        m_t0 = time.perf_counter()
        # Fold-safe CV pipeline: SMOTE runs strictly inside each CV fold!
        cv_pipe = ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.0, random_state=RANDOM_SEED)),
            ("model", model),
        ])
        aucs = cross_val_score(cv_pipe, X_cv, y_cv, cv=skf, scoring="roc_auc", n_jobs=1)

        # Fit final model on full SMOTE-balanced training partition
        model.fit(X_train_smote, y_train_smote)
        trained_classifiers[name] = model
        elapsed_m = round(time.perf_counter() - m_t0, 2)
        cv_scores[name] = {
            "cv_roc_auc_mean": round(float(np.mean(aucs)), 4),
            "cv_roc_auc_std": round(float(np.std(aucs)), 4),
            "train_time_sec": elapsed_m,
        }
        print(f"  -> Trained {name:30s} | 3-Fold CV ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f}) [{elapsed_m}s]")

    # Train Secondary Baseline: Continuous Delay Length Regressor (in minutes)
    reg_t0 = time.perf_counter()
    delay_regressor = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    delay_regressor.fit(X_train_raw, y_train_reg)
    reg_elapsed = round(time.perf_counter() - reg_t0, 2)
    print(f"  -> Trained Secondary XGBoost Delay Regressor (minutes) [{reg_elapsed}s]")

    bundle = {
        "classifiers": trained_classifiers,
        "primary_model_name": "XGBoost Classifier (Tuned)",
        "delay_regressor": delay_regressor,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_metadata": feature_meta,
        "cv_scores": cv_scores,
        "default_threshold": 0.50,
    }
    model_path = MODELS_DIR / "flight_delay_models.joblib"
    joblib.dump(bundle, model_path)

    train_summary = {
        "models_trained": list(trained_classifiers.keys()),
        "secondary_regressor": "XGBRegressor (Delay Minutes)",
        "cv_results": cv_scores,
        "model_bundle_path": str(model_path),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(OUTPUTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    return train_summary


if __name__ == "__main__":
    run_train_models()
