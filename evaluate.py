"""
Stage 6 — 1M+ Row Untouched Test Evaluation, Threshold Tuning & Plot Export (evaluate.py)
Module Owner: Abhijeet Rana (S24CSEU1419)

Responsibilities:
  1. Evaluates all trained classifiers across the FULL 1,000,000+ row Untouched Chronological
     Test Split (Oct–Dec 2024 real U.S. flights) using ROC-AUC, PR-AUC, Precision, Recall, F1, & F2.
  2. Calibrates operational decision threshold on the held-out training validation slice (zero test leakage).
  3. Exports publication-grade plots to outputs/plots/ and compact summary artifacts for Streamlit.
"""

from __future__ import annotations
import json
import time
import duckdb
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
)

from config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, MODEL_FEATURE_COLUMNS

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

FEATURE_DISPLAY_NAMES = {
    "DISTANCE": "Flight Distance (mi)",
    "CRS_ELAPSED_TIME": "Scheduled Block Time (min)",
    "dep_hour_sin": "Departure Hour (Sin)",
    "dep_hour_cos": "Departure Hour (Cos)",
    "dow_sin": "Day of Week (Sin)",
    "dow_cos": "Day of Week (Cos)",
    "month_sin": "Seasonal Month (Sin)",
    "month_cos": "Seasonal Month (Cos)",
    "is_weekend": "Weekend Indicator",
    "TEMP_F": "Origin Temp (°F)",
    "DEW_POINT_F": "Origin Dew Point (°F)",
    "temp_dew_spread": "Temp-Dew Fog Spread (°F)",
    "WIND_SPEED_KT": "Surface Wind Speed (kt)",
    "VISIBILITY_MI": "Surface Visibility (mi)",
    "PRECIP_IN": "Hourly Precip (in/hr)",
    "PRESSURE_MB": "Sea-Level Pressure (mb)",
    "wind_max_3h": "3h Peak Wind Gust/Speed (kt)",
    "precip_sum_3h": "3h Cumulative Precip (in)",
    "vis_min_3h": "3h Minimum Visibility (mi)",
    "origin_rolling_3h_vol": "Origin 3h Rolling Dep. Volume",
    "origin_congestion_ratio": "Origin Hub Congestion Ratio",
    "prev_aircraft_delay_t2h": "Prev-Leg Aircraft Delay @ T-2h (min)",
    "has_inbound_leg": "Has Same-Day Inbound Aircraft Leg",
    "origin_t2h_delay_rate": "Origin Hub Delay Rate @ T-2h",
    "dest_t2h_delay_rate": "Dest. Hub Delay Rate @ T-2h",
    "carrier_hist_delay_rate": "Carrier Historical Delay Rate",
    "origin_hist_delay_rate": "Origin Hub Historical Delay Rate",
    "dest_hist_delay_rate": "Dest. Hub Historical Delay Rate",
}


def run_evaluation() -> dict:
    t0 = time.perf_counter()
    splits = joblib.load(PROCESSED_DIR / "train_test_splits.joblib")
    bundle = joblib.load(MODELS_DIR / "flight_delay_models.joblib")
    with open(OUTPUTS_DIR / "smote_summary.json", "r", encoding="utf-8") as f:
        smote_info = json.load(f)

    cutoff_ts = splits["cutoff_ts"]
    feat_parquet = PROCESSED_DIR / "featured_flights.parquet"
    feat_cols_sql = ", ".join(MODEL_FEATURE_COLUMNS)

    # Stream full 1M+ untouched chronological test set cleanly as float32 (~110 MB RAM)
    con = duckdb.connect()
    try:
        test_df = con.execute(
            f"""
            SELECT {feat_cols_sql}, DEP_DEL15, DEP_DELAY
            FROM read_parquet('{feat_parquet.as_posix()}')
            WHERE CRS_DEP_TIMESTAMP >= '{cutoff_ts}'
            """
        ).fetchdf()
    finally:
        con.close()

    X_test = test_df[MODEL_FEATURE_COLUMNS].astype(np.float32)
    for int_col in [
        "DISTANCE", "CRS_ELAPSED_TIME", "is_weekend", "has_inbound_leg",
        "TEMP_F", "DEW_POINT_F", "WIND_SPEED_KT", "wind_max_3h",
        "origin_rolling_3h_vol", "prev_aircraft_delay_t2h",
    ]:
        X_test[int_col] = np.round(X_test[int_col]).astype(np.float32)
    X_test["temp_dew_spread"] = (X_test["TEMP_F"] - X_test["DEW_POINT_F"]).astype(np.float32)
    X_test["VISIBILITY_MI"] = (np.round(X_test["VISIBILITY_MI"] * 2.0) / 2.0).astype(np.float32)
    X_test["vis_min_3h"] = (np.round(X_test["vis_min_3h"] * 2.0) / 2.0).astype(np.float32)
    X_test["PRESSURE_MB"] = np.round(X_test["PRESSURE_MB"], 0).astype(np.float32)
    X_test["PRECIP_IN"] = np.round(X_test["PRECIP_IN"], 2).astype(np.float32)
    X_test["precip_sum_3h"] = np.round(X_test["precip_sum_3h"], 2).astype(np.float32)
    y_test = test_df["DEP_DEL15"].to_numpy(dtype=np.int32)
    y_test_reg = np.clip(test_df["DEP_DELAY"].to_numpy(dtype=np.float32), -30.0, 360.0)
    del test_df

    classifiers = bundle["classifiers"]
    regressor = bundle["delay_regressor"]
    feature_cols = bundle["feature_columns"]

    # Calibrate F1-optimal operational threshold strictly on held-out training calibration fold
    X_cal = splits["X_cal"]
    y_cal = splits["y_cal"].to_numpy()
    xgb_cal_probs = classifiers["XGBoost Classifier (Tuned)"].predict_proba(X_cal)[:, 1]
    best_th, best_score = 0.55, -1.0
    for cand_th in np.arange(0.38, 0.74, 0.02):
        cand_f = f1_score(y_cal, (xgb_cal_probs >= cand_th).astype(int), zero_division=0)
        if cand_f > best_score:
            best_score = cand_f
            best_th = round(float(cand_th), 2)

    bundle["default_threshold"] = best_th
    joblib.dump(bundle, MODELS_DIR / "flight_delay_models.joblib", compress=3)

    benchmarks = []
    curves_data = {}
    confusion_data = {}
    xgb_probs_full = None

    print(f"[evaluate.py] Scoring all 4 models on {len(y_test):,} untouched late-2024 chronological test flights...")
    for name, clf in classifiers.items():
        cal_p = clf.predict_proba(X_cal)[:, 1]
        m_th = max(
            np.arange(0.38, 0.74, 0.02),
            key=lambda t: f1_score(y_cal, (cal_p >= t).astype(int), zero_division=0),
        )
        probs = clf.predict_proba(X_test)[:, 1]
        if "XGBoost" in name:
            xgb_probs_full = probs
            m_th = best_th
        preds = (probs >= float(m_th)).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        f2 = fbeta_score(y_test, preds, beta=2.0, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)

        fpr, tpr, _ = roc_curve(y_test, probs)
        pr_p, pr_r, _ = precision_recall_curve(y_test, probs)

        idx_roc = np.linspace(0, len(fpr) - 1, min(220, len(fpr)), dtype=int)
        idx_pr = np.linspace(0, len(pr_p) - 1, min(220, len(pr_p)), dtype=int)

        curves_data[name] = {
            "fpr": np.round(fpr[idx_roc], 4).tolist(),
            "tpr": np.round(tpr[idx_roc], 4).tolist(),
            "pr_precision": np.round(pr_p[idx_pr], 4).tolist(),
            "pr_recall": np.round(pr_r[idx_pr], 4).tolist(),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
        }
        confusion_data[name] = cm.tolist()

        benchmarks.append({
            "Model Architecture": name,
            "Accuracy": round(float(acc * 100.0), 2),
            "Precision": round(float(prec), 3),
            "Recall": round(float(rec), 3),
            "F1-Score": round(float(f1), 3),
            "F2-Score": round(float(f2), 3),
            "ROC-AUC": round(float(roc_auc), 3),
            "PR-AUC": round(float(pr_auc), 3),
        })

    benchmarks_df = pd.DataFrame(benchmarks)
    benchmarks_df.to_csv(OUTPUTS_DIR / "benchmark_table.csv", index=False)

    # Secondary Continuous Delay Regressor evaluation
    reg_preds = regressor.predict(X_test)
    reg_mae = float(mean_absolute_error(y_test_reg, reg_preds))
    reg_rmse = float(root_mean_squared_error(y_test_reg, reg_preds))

    # Threshold Tuning Sweep across full 1M+ test set on Tuned XGBoost
    thresholds = np.round(np.arange(0.15, 0.86, 0.025), 3)
    thresh_rows = []
    for th in thresholds:
        p_th = (xgb_probs_full >= th).astype(int)
        cm_th = confusion_matrix(y_test, p_th)
        thresh_rows.append({
            "threshold": float(th),
            "accuracy": round(float(accuracy_score(y_test, p_th)), 4),
            "precision": round(float(precision_score(y_test, p_th, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, p_th, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, p_th, zero_division=0)), 4),
            "f2_ops": round(float(fbeta_score(y_test, p_th, beta=2.0, zero_division=0)), 4),
            "tn": int(cm_th[0, 0]),
            "fp": int(cm_th[0, 1]),
            "fn": int(cm_th[1, 0]),
            "tp": int(cm_th[1, 1]),
        })
    thresh_df = pd.DataFrame(thresh_rows)
    best_f1_row = thresh_df.loc[thresh_df["f1"].idxmax()].to_dict()
    best_f2_row = thresh_df.loc[thresh_df["f2_ops"].idxmax()].to_dict()

    # Feature Importances + Correlation Rankings
    xgb_clf = classifiers["XGBoost Classifier (Tuned)"]
    importances = xgb_clf.feature_importances_
    corr_sample = min(200000, len(y_test))
    corrs = [float(np.corrcoef(X_test[c].to_numpy()[:corr_sample], y_test[:corr_sample])[0, 1]) for c in feature_cols]
    feat_df = pd.DataFrame({
        "feature": feature_cols,
        "display_name": [FEATURE_DISPLAY_NAMES.get(c, c) for c in feature_cols],
        "xgb_importance": np.round(importances, 4),
        "target_correlation": np.round(corrs, 4),
    }).sort_values("xgb_importance", ascending=False).reset_index(drop=True)
    feat_df.to_csv(OUTPUTS_DIR / "feature_importance_rankings.csv", index=False)

    # Save stratified 20,000-row sample of test predictions for instant Confusion Matrix slider in Streamlit
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(y_test), size=min(20000, len(y_test)), replace=False)
    np.savez_compressed(
        OUTPUTS_DIR / "xgb_test_predictions.npz",
        y_true=y_test[sample_idx],
        y_prob=xgb_probs_full[sample_idx],
        full_count=len(y_test),
    )

    # ---------------- EXPORT PLOTS ----------------
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for ax, (name, cm) in zip(axes.flatten(), confusion_data.items()):
        cm_arr = np.array(cm)
        sns.heatmap(
            cm_arr, annot=True, fmt=",d", cmap="Blues", cbar=False,
            xticklabels=["Pred On-Time (0)", "Pred Delayed (1)"],
            yticklabels=["Actual On-Time (0)", "Actual Delayed (1)"],
            ax=ax, annot_kws={"size": 11.5, "weight": "bold"},
        )
        row_m = next(r for r in benchmarks if r["Model Architecture"] == name)
        ax.set_title(
            f"{name}\nROC-AUC: {row_m['ROC-AUC']:.3f} | Recall: {row_m['Recall']:.2f} | Precision: {row_m['Precision']:.2f}",
            fontsize=11, weight="bold",
        )
    fig.suptitle(f"Untouched Chronological Test Split (n = {len(y_test):,} Real Flights) — Confusion Matrices", fontsize=13.5, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    palette = ["#64748b", "#2563eb", "#0d9488", "#dc2626"]
    for (name, cdata), color in zip(curves_data.items(), palette):
        lw = 2.8 if "XGBoost" in name else 2.0
        ax1.plot(cdata["fpr"], cdata["tpr"], label=f"{name} (AUC = {cdata['roc_auc']:.3f})", color=color, lw=lw)
        ax2.plot(cdata["pr_recall"], cdata["pr_precision"], label=f"{name} (AP = {cdata['pr_auc']:.3f})", color=color, lw=lw)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance (0.500)")
    ax1.set_title(f"ROC Curves ({len(y_test):,} Untouched Real 2024 Flights)", fontsize=12, weight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right", fontsize=9)

    baseline_pos = float(y_test.mean())
    ax2.axhline(baseline_pos, color="k", linestyle="--", alpha=0.4, label=f"Operational Delay Rate ({baseline_pos:.2f})")
    ax2.set_title("Precision-Recall Curves (Real BTS Class Skew)", fontsize=12, weight="bold")
    ax2.set_xlabel("Recall (Delay Detection Rate)")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_pr_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall (Delay Catch Rate)", color="#dc2626", lw=2.5)
    ax.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color="#2563eb", lw=2.5)
    ax.plot(thresh_df["threshold"], thresh_df["f1"], label="F1-Score", color="#0f766e", lw=2.3)
    ax.plot(thresh_df["threshold"], thresh_df["f2_ops"], label="F2-Score (Recall-Prioritized Ops)", color="#7c3aed", lw=2.2, linestyle="--")
    ax.axvline(best_th, color="#0f172a", alpha=0.6, linestyle=":", label=f"Calibrated Cutoff ({best_th:.2f})")
    ax.set_title("XGBoost Operational Threshold Tuning — Precision vs. Recall Trade-off", fontsize=12, weight="bold")
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Metric Score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "threshold_tuning.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    top_f = feat_df.head(16).iloc[::-1]
    ax.barh(top_f["display_name"], top_f["xgb_importance"], color="#1d4ed8", alpha=0.88)
    ax.set_title("Tuned XGBoost Feature Importance Rankings (2024 U.S. BTS + NOAA ASOS)", fontsize=12, weight="bold")
    ax.set_xlabel("Relative Gain Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    panels = [
        ("Full 80% Train Partition", smote_info["full_train_partition"]["on_time_class_0"], smote_info["full_train_partition"]["delayed_class_1"]),
        ("SMOTE Balanced Train Fold", smote_info["train_after_smote"]["on_time_class_0"], smote_info["train_after_smote"]["delayed_class_1"]),
        ("Untouched 20% Test Split", smote_info["test_untouched"]["on_time_class_0"], smote_info["test_untouched"]["delayed_class_1"]),
    ]
    for ax, (title, c0, c1) in zip(axes, panels):
        bars = ax.bar(["On-Time (<15m)", "Delayed (>=15m)"], [c0, c1], color=["#0ea5e9", "#ef4444"], width=0.55)
        ax.set_title(title, fontsize=10.5, weight="bold")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f"{int(b.get_height()):,}", ha="center", fontsize=9, weight="bold")
        ax.set_ylim(0, max(c0, c1) * 1.18)
    fig.suptitle("Multi-Million-Row Chronological Partitioning & Training-Fold SMOTE", fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance_smote.png")
    plt.close(fig)

    # Free temporary training split cache after model & evaluation completion to save disk space
    (PROCESSED_DIR / "train_test_splits.joblib").unlink(missing_ok=True)

    evaluation_report = {
        "test_rows_evaluated": int(len(y_test)),
        "calibrated_threshold": best_th,
        "benchmarks": benchmarks,
        "secondary_regression_baseline": {
            "model": "XGBoost Delay Length Regressor (Minutes)",
            "mae_minutes": round(reg_mae, 2),
            "rmse_minutes": round(reg_rmse, 2),
        },
        "confusion_matrices": confusion_data,
        "curves": curves_data,
        "threshold_tuning": thresh_rows,
        "optimal_f1_threshold": best_f1_row,
        "optimal_f2_ops_threshold": best_f2_row,
        "feature_rankings": feat_df.to_dict(orient="records"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    with open(OUTPUTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n=== Untouched 20% Chronological Test Split ({len(y_test):,} Real 2024 Flights) Benchmarks ===")
    print(benchmarks_df.to_string(index=False))
    print(f"\nSecondary Regression Baseline -> MAE: {reg_mae:.2f} min | RMSE: {reg_rmse:.2f} min")
    return evaluation_report


if __name__ == "__main__":
    run_evaluation()
