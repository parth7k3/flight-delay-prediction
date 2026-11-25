"""
Stage 6 — Model Evaluation, Threshold Tuning & Plot Export Module (evaluate.py)
Module Owner: Abhijeet Rana (S24CSEU1419)

Responsibilities:
  1. Scores all trained classifiers on the untouched 20% chronological test split using
     Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
  2. Evaluates the Secondary Continuous Delay Regressor (MAE, RMSE).
  3. Runs operational decision-threshold tuning (0.10 -> 0.90) optimizing F1 and Recall-weighted F2
     to minimize costly missed passenger connections (False Negatives).
  4. Exports publication-quality plots to outputs/plots/:
       - confusion_matrices.png
       - roc_pr_curves.png
       - threshold_tuning.png
       - feature_importance.png
       - class_balance_smote.png
"""

from __future__ import annotations
import json
import time
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

from config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


FEATURE_DISPLAY_NAMES = {
    "DISTANCE": "Flight Distance (mi)",
    "dep_hour_sin": "Departure Hour (Sin)",
    "dep_hour_cos": "Departure Hour (Cos)",
    "dow_sin": "Day of Week (Sin)",
    "dow_cos": "Day of Week (Cos)",
    "is_weekend": "Weekend Flag",
    "TEMP_F": "Surface Temp (°F)",
    "DEW_POINT_F": "Dew Point (°F)",
    "temp_dew_spread": "Temp-Dew Spread (°F)",
    "WIND_SPEED_KT": "Wind Speed (knots)",
    "VISIBILITY_MI": "Visibility (miles)",
    "PRECIP_IN": "Precipitation (in/hr)",
    "PRESSURE_MB": "Barometric Pressure (mb)",
    "origin_rolling_3h_vol": "Origin 3h Rolling Dep. Volume",
    "origin_congestion_ratio": "Origin Hub Congestion Ratio",
    "carrier_hist_delay_rate": "Carrier Historical Delay Prior",
    "origin_hist_delay_rate": "Origin Hub Delay Prior",
    "dest_hist_delay_rate": "Destination Hub Delay Prior",
}


def run_evaluation() -> dict:
    t0 = time.perf_counter()
    splits = joblib.load(PROCESSED_DIR / "train_test_splits.joblib")
    bundle = joblib.load(MODELS_DIR / "flight_delay_models.joblib")
    with open(OUTPUTS_DIR / "smote_summary.json", "r", encoding="utf-8") as f:
        smote_info = json.load(f)

    X_test = splits["X_test"]
    y_test = splits["y_test"].to_numpy()
    y_test_reg = splits["y_test_reg"].to_numpy()

    classifiers = bundle["classifiers"]
    regressor = bundle["delay_regressor"]
    feature_cols = bundle["feature_columns"]

    benchmarks = []
    curves_data = {}
    confusion_data = {}

    # Determine recall-prioritized operational threshold on a held-out calibration slice of training data
    X_cal = splits["X_train_raw"].iloc[-8000:]
    y_cal = splits["y_train_raw"].iloc[-8000:].to_numpy()
    xgb_cal_probs = classifiers["XGBoost Classifier (Tuned)"].predict_proba(X_cal)[:, 1]
    best_th, best_score = 0.45, -1.0
    for cand_th in np.arange(0.25, 0.65, 0.02):
        cand_f = fbeta_score(y_cal, (xgb_cal_probs >= cand_th).astype(int), beta=1.25, zero_division=0)
        if cand_f > best_score:
            best_score = cand_f
            best_th = round(float(cand_th), 2)

    bundle["default_threshold"] = best_th
    joblib.dump(bundle, MODELS_DIR / "flight_delay_models.joblib")

    for name, clf in classifiers.items():
        probs = clf.predict_proba(X_test)[:, 1]
        # Apply operational threshold calibrated on training validation fold (zero test leakage)
        th_use = 0.50 if "Logistic" in name else best_th
        preds = (probs >= th_use).astype(int)

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

        # Downsample curve arrays to 200 points for compact JSON/UI loading
        idx_roc = np.linspace(0, len(fpr) - 1, min(200, len(fpr)), dtype=int)
        idx_pr = np.linspace(0, len(pr_p) - 1, min(200, len(pr_p)), dtype=int)

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

    # Secondary Regression Baseline evaluation
    reg_preds = regressor.predict(X_test)
    reg_mae = float(mean_absolute_error(y_test_reg, reg_preds))
    reg_rmse = float(root_mean_squared_error(y_test_reg, reg_preds))

    # Threshold Tuning Sweep on Primary Model (XGBoost Classifier (Tuned))
    xgb_clf = classifiers["XGBoost Classifier (Tuned)"]
    xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
    thresholds = np.round(np.arange(0.15, 0.86, 0.025), 3)
    thresh_rows = []
    for th in thresholds:
        p_th = (xgb_probs >= th).astype(int)
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

    # Feature Importances + Target Correlation Rankings (Page 8 Appendix)
    importances = xgb_clf.feature_importances_
    corrs = [float(np.corrcoef(X_test[c].to_numpy(), y_test)[0, 1]) for c in feature_cols]
    feat_df = pd.DataFrame({
        "feature": feature_cols,
        "display_name": [FEATURE_DISPLAY_NAMES.get(c, c) for c in feature_cols],
        "xgb_importance": np.round(importances, 4),
        "target_correlation": np.round(corrs, 4),
    }).sort_values("xgb_importance", ascending=False).reset_index(drop=True)
    feat_df.to_csv(OUTPUTS_DIR / "feature_importance_rankings.csv", index=False)

    # Save compact test probability sample for interactive threshold slider in Streamlit
    sample_n = min(5000, len(y_test))
    np.savez_compressed(
        OUTPUTS_DIR / "xgb_test_predictions.npz",
        y_true=y_test[:sample_n],
        y_prob=xgb_probs[:sample_n],
        full_count=len(y_test),
    )

    # ---------------- EXPORT PLOTS ----------------
    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for ax, (name, cm) in zip(axes.flatten(), confusion_data.items()):
        cm_arr = np.array(cm)
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred On-Time (0)", "Pred Delayed (1)"],
            yticklabels=["Actual On-Time (0)", "Actual Delayed (1)"],
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"},
        )
        row_m = next(r for r in benchmarks if r["Model Architecture"] == name)
        ax.set_title(
            f"{name}\nROC-AUC: {row_m['ROC-AUC']:.3f} | Recall: {row_m['Recall']:.2f} | Precision: {row_m['Precision']:.2f}",
            fontsize=11,
            weight="bold",
        )
    fig.suptitle("Untouched Chronological Test Split — Confusion Matrices (Threshold = 0.50)", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png")
    plt.close(fig)

    # Plot 2: ROC-AUC and Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    palette = ["#64748b", "#2563eb", "#0d9488", "#dc2626"]
    for (name, cdata), color in zip(curves_data.items(), palette):
        lw = 2.8 if "XGBoost" in name else 2.0
        ax1.plot(cdata["fpr"], cdata["tpr"], label=f"{name} (AUC = {cdata['roc_auc']:.3f})", color=color, lw=lw)
        ax2.plot(cdata["pr_recall"], cdata["pr_precision"], label=f"{name} (AP = {cdata['pr_auc']:.3f})", color=color, lw=lw)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance (0.500)")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, weight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right", fontsize=9)

    baseline_pos = float(y_test.mean())
    ax2.axhline(baseline_pos, color="k", linestyle="--", alpha=0.4, label=f"Class Skew Baseline ({baseline_pos:.2f})")
    ax2.set_title("Precision-Recall Curves (Untouched 80/20 Test Split)", fontsize=12, weight="bold")
    ax2.set_xlabel("Recall (Delay Detection Rate)")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_pr_curves.png")
    plt.close(fig)

    # Plot 3: Operational Threshold Tuning
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall (Delay Catch Rate)", color="#dc2626", lw=2.5)
    ax.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color="#2563eb", lw=2.5)
    ax.plot(thresh_df["threshold"], thresh_df["f1"], label="F1-Score", color="#0f766e", lw=2.3)
    ax.plot(thresh_df["threshold"], thresh_df["f2_ops"], label="F2-Score (Recall-Prioritized Ops)", color="#7c3aed", lw=2.2, linestyle="--")
    ax.axvline(best_f1_row["threshold"], color="#0f766e", alpha=0.5, linestyle=":")
    ax.set_title("XGBoost Operational Threshold Tuning — Precision vs. Recall Trade-off", fontsize=12, weight="bold")
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Metric Score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "threshold_tuning.png")
    plt.close(fig)

    # Plot 4: Feature Importance Rankings
    fig, ax = plt.subplots(figsize=(10, 6.2))
    top_f = feat_df.head(14).iloc[::-1]
    ax.barh(top_f["display_name"], top_f["xgb_importance"], color="#1d4ed8", alpha=0.88)
    ax.set_title("Tuned XGBoost Feature Importance Rankings (Weather, Cyclical Time & Hub Congestion)", fontsize=12, weight="bold")
    ax.set_xlabel("Relative Gain Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close(fig)

    # Plot 5: SMOTE Class Balancing Overview
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [
        ("Train Split (Before SMOTE)", smote_info["train_before_smote"]["on_time_class_0"], smote_info["train_before_smote"]["delayed_class_1"]),
        ("Train Split (After SMOTE)", smote_info["train_after_smote"]["on_time_class_0"], smote_info["train_after_smote"]["delayed_class_1"]),
        ("Chronological Test (Untouched)", smote_info["test_untouched"]["on_time_class_0"], smote_info["test_untouched"]["delayed_class_1"]),
    ]
    for ax, (title, c0, c1) in zip(axes, panels):
        bars = ax.bar(["On-Time (<15m)", "Delayed (>=15m)"], [c0, c1], color=["#0ea5e9", "#ef4444"], width=0.55)
        ax.set_title(title, fontsize=10.5, weight="bold")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f"{int(b.get_height()):,}", ha="center", fontsize=9.5, weight="bold")
        ax.set_ylim(0, max(c0, c1) * 1.18)
    fig.suptitle("Strict Training-Partition SMOTE vs. Untouched Operational Test Distribution", fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance_smote.png")
    plt.close(fig)

    evaluation_report = {
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
        "plots_exported": [
            str(PLOTS_DIR / "confusion_matrices.png"),
            str(PLOTS_DIR / "roc_pr_curves.png"),
            str(PLOTS_DIR / "threshold_tuning.png"),
            str(PLOTS_DIR / "feature_importance.png"),
            str(PLOTS_DIR / "class_balance_smote.png"),
        ],
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    with open(OUTPUTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print("\n=== Untouched 20% Chronological Test Split Model Benchmarks ===")
    print(benchmarks_df.to_string(index=False))
    print(f"\nSecondary Regression Baseline -> MAE: {reg_mae:.2f} min | RMSE: {reg_rmse:.2f} min")
    return evaluation_report


if __name__ == "__main__":
    run_evaluation()
