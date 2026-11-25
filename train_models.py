"""
Stage 5 — Scalable Big Data Ensemble Training & Cross-Validation (train_models.py)
Module Owner: Krishna Sharma (S24CSEU2756)

Responsibilities:
  1. Runs 3-fold fold-internal SMOTE cross-validation to guard against synthetic leakage.
  2. Trains four primary classification models on the ~480,000 SMOTE-balanced real flight vectors:
       - Logistic Regression (Baseline)
       - Random Forest (100 Trees)
       - Gradient Boosting (Histogram-based scalable GBDT)
       - XGBoost Classifier (Tuned, Histogram tree_method='hist')
  3. Trains Secondary Continuous Regression Baseline (XGBoost Regressor) on real departure delay minutes.
"""

from __future__ import annotations
import json
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

    classifiers = {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest (100 Trees)": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=25,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.06,
            max_depth=8,
            min_samples_leaf=40,
            l2_regularization=1.5,
            random_state=RANDOM_SEED,
        ),
        "XGBoost Classifier (Tuned)": XGBClassifier(
            tree_method="hist",
            n_estimators=320,
            learning_rate=0.055,
            max_depth=8,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.4,
            reg_lambda=2.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }

    # 3-Fold Fold-Internal SMOTE Cross-Validation on 40,000 real training rows
    cv_n = min(40000, len(X_train_raw))
    X_cv = X_train_raw.iloc[:cv_n]
    y_cv = y_train_raw.iloc[:cv_n]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    cv_scores = {}
    trained_classifiers = {}

    for name, model in classifiers.items():
        m_t0 = time.perf_counter()
        cv_pipe = ImbPipeline([
            ("smote", SMOTE(sampling_strategy=1.0, random_state=RANDOM_SEED)),
            ("model", model),
        ])
        aucs = cross_val_score(cv_pipe, X_cv, y_cv, cv=skf, scoring="roc_auc", n_jobs=1)

        model.fit(X_train_smote, y_train_smote)
        trained_classifiers[name] = model
        elapsed_m = round(time.perf_counter() - m_t0, 2)
        cv_scores[name] = {
            "cv_roc_auc_mean": round(float(np.mean(aucs)), 4),
            "cv_roc_auc_std": round(float(np.std(aucs)), 4),
            "train_time_sec": elapsed_m,
        }
        print(f"  -> Trained {name:30s} | 3-Fold CV ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f}) [{elapsed_m}s]")

    # Secondary Continuous Regression Baseline (clipped extreme outliers >360m during fit for stable MAE)
    reg_t0 = time.perf_counter()
    delay_regressor = XGBRegressor(
        tree_method="hist",
        n_estimators=220,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=25,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    delay_regressor.fit(X_train_raw, np.clip(y_train_reg, -25.0, 300.0))
    reg_elapsed = round(time.perf_counter() - reg_t0, 2)
    print(f"  -> Trained Secondary XGBoost Delay Regressor (minutes) [{reg_elapsed}s]")

    bundle = {
        "classifiers": trained_classifiers,
        "primary_model_name": "XGBoost Classifier (Tuned)",
        "delay_regressor": delay_regressor,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_metadata": feature_meta,
        "cv_scores": cv_scores,
        "default_threshold": 0.45,
    }
    model_path = MODELS_DIR / "flight_delay_models.joblib"
    joblib.dump(bundle, model_path, compress=3)

    train_summary = {
        "models_trained": list(trained_classifiers.keys()),
        "secondary_regressor": "XGBRegressor (Delay Minutes, tree_method='hist')",
        "smote_training_vectors": int(len(y_train_smote)),
        "cv_results": cv_scores,
        "model_bundle_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(OUTPUTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    return train_summary


if __name__ == "__main__":
    run_train_models()
