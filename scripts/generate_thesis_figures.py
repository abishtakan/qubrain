from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qubrain.scripts.train_model import load_aligned_dataset

ARTIFACT_DIR = PROJECT_ROOT / "qubrain" / "backend" / "model_artifacts"
OUTPUT_DIR = PROJECT_ROOT / "qubrain" / "plots" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> Path:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def load_classifier_artifacts() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    holdout = pd.read_csv(ARTIFACT_DIR / "holdout_predictions.csv")
    cv_results = pd.read_csv(ARTIFACT_DIR / "cv_results.csv")
    baseline = pd.read_csv(ARTIFACT_DIR / "baseline_benchmark.csv")
    explainability = json.loads((ARTIFACT_DIR / "explainability.json").read_text(encoding="utf-8"))
    return metadata, holdout, cv_results, baseline, explainability


def build_curve_inputs(holdout: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_true = (holdout["actual_status"] == "Dead").astype(int).to_numpy()
    y_score = holdout["mortality_probability"].to_numpy(dtype=float)
    return y_true, y_score


def plot_class_distribution(metadata: dict[str, object]) -> Path:
    summary = metadata["dataset_summary"]["final_labeled_cohort"]
    labels = ["Dead", "Alive"]
    values = [summary["dead"], summary["alive"]]
    colors = ["#b22222", "#2f6bff"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Cohort Class Distribution")
    plt.ylabel("Patient Count")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center")
    return save_figure("01_cohort_class_distribution.png")


def plot_split_distribution(metadata: dict[str, object]) -> Path:
    split = metadata["dataset_summary"]["split_summary"]
    categories = ["Dead", "Alive"]
    train_values = [split["train_dead"], split["train_alive"]]
    test_values = [split["holdout_dead"], split["holdout_alive"]]
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, train_values, width=width, label="Train", color="#4c78a8")
    plt.bar(x + width / 2, test_values, width=width, label="Holdout", color="#f58518")
    plt.xticks(x, categories)
    plt.ylabel("Patient Count")
    plt.title("Train vs Holdout Class Distribution")
    plt.legend()
    return save_figure("02_train_holdout_distribution.png")


def plot_age_distribution(dataset) -> Path:
    df = pd.DataFrame(
        {
            "age": dataset.age,
            "status": np.where(dataset.labels == 1, "Dead", "Alive"),
        }
    )
    groups = [df.loc[df["status"] == "Alive", "age"], df.loc[df["status"] == "Dead", "age"]]

    plt.figure(figsize=(7, 4))
    plt.boxplot(groups, tick_labels=["Alive", "Dead"], patch_artist=True)
    plt.ylabel("Age")
    plt.title("Age Distribution by Mortality Status")
    return save_figure("03_age_distribution_by_status.png")


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#d62728", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Holdout ROC Curve")
    plt.legend(loc="lower right")
    return save_figure("04_holdout_roc_curve.png")


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#1f77b4", linewidth=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Holdout Precision-Recall Curve")
    plt.legend(loc="lower left")
    return save_figure("05_holdout_pr_curve.png")


def plot_confusion_matrix(holdout: pd.DataFrame, threshold: float) -> Path:
    y_true = (holdout["actual_status"] == "Dead").astype(int).to_numpy()
    y_pred = (holdout["mortality_probability"].to_numpy(dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Pred Alive", "Pred Dead"])
    plt.yticks([0, 1], ["Actual Alive", "Actual Dead"])
    plt.title("Holdout Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    return save_figure("06_holdout_confusion_matrix.png")


def plot_probability_histogram(holdout: pd.DataFrame) -> Path:
    alive_probs = holdout.loc[holdout["actual_status"] == "Alive", "mortality_probability"].to_numpy(dtype=float)
    dead_probs = holdout.loc[holdout["actual_status"] == "Dead", "mortality_probability"].to_numpy(dtype=float)

    plt.figure(figsize=(7, 4.5))
    plt.hist(alive_probs, bins=12, alpha=0.7, label="Alive", color="#4c78a8")
    plt.hist(dead_probs, bins=12, alpha=0.7, label="Dead", color="#e45756")
    plt.xlabel("Predicted Mortality Probability")
    plt.ylabel("Patient Count")
    plt.title("Holdout Probability Distribution")
    plt.legend()
    return save_figure("07_holdout_probability_distribution.png")


def plot_calibration_curve(y_true: np.ndarray, y_score: np.ndarray) -> Path:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=6, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.plot(mean_pred, frac_pos, marker="o", color="#2ca02c", linewidth=2)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title("Holdout Calibration Curve")
    return save_figure("08_holdout_calibration_curve.png")


def plot_threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> Path:
    thresholds = np.linspace(0.1, 0.9, 81)
    balanced = []
    f1_values = []

    for threshold in thresholds:
        preds = (y_score >= threshold).astype(int)
        balanced.append(balanced_accuracy_score(y_true, preds))
        f1_values.append(f1_score(y_true, preds, zero_division=0))

    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, balanced, label="Balanced Accuracy", color="#9467bd", linewidth=2)
    plt.plot(thresholds, f1_values, label="F1 Score", color="#ff7f0e", linewidth=2)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep on Holdout Set")
    plt.legend()
    return save_figure("09_threshold_sweep.png")


def plot_cv_scatter(cv_results: pd.DataFrame) -> Path:
    plt.figure(figsize=(6.5, 5))
    colors = cv_results["imbalance_strategy"].map({"class_weight": "#4c78a8", "smote": "#f58518"}).fillna("#999999")
    plt.scatter(
        cv_results["mean_val_auc"],
        cv_results["mean_val_balanced_accuracy"],
        s=90,
        c=colors,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in cv_results.iterrows():
        plt.text(row["mean_val_auc"] + 0.001, row["mean_val_balanced_accuracy"] + 0.001, f"q{int(row['n_qubits'])}/l{int(row['n_layers'])}", fontsize=8)
    plt.xlabel("Mean Validation AUC")
    plt.ylabel("Mean Validation Balanced Accuracy")
    plt.title("Cross-Validation Candidate Comparison")
    return save_figure("10_cv_auc_vs_balanced_accuracy.png")


def plot_cv_overfit(cv_results: pd.DataFrame) -> Path:
    ranked = cv_results.sort_values("mean_val_auc", ascending=False).reset_index(drop=True)
    labels = [f"{row.n_qubits}q-{row.n_layers}l-{row.imbalance_strategy}" for row in ranked.itertuples()]

    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, ranked["mean_overfit_gap_auc"], color="#72b7b2")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Train AUC - Validation AUC")
    plt.title("Cross-Validation Overfitting Gap")
    return save_figure("11_cv_overfit_gap.png")


def plot_baseline_comparison(baseline: pd.DataFrame, metadata: dict[str, object]) -> Path:
    model_rows = baseline[["model", "mean_auc", "mean_balanced_accuracy"]].copy()
    hybrid_row = pd.DataFrame(
        [
            {
                "model": "Hybrid Quantum-Classical",
                "mean_auc": metadata["train_cv_auc"],
                "mean_balanced_accuracy": metadata["inner_cv_balanced_accuracy"],
            }
        ]
    )
    df = pd.concat([model_rows, hybrid_row], ignore_index=True)
    df = df.sort_values("mean_auc", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df["model"], df["mean_auc"], color="#4c78a8")
    plt.xlabel("Mean AUC")
    plt.title("Model Comparison by Cross-Validated AUC")
    return save_figure("12_model_comparison_auc.png")


def plot_baseline_balanced_accuracy(baseline: pd.DataFrame, metadata: dict[str, object]) -> Path:
    model_rows = baseline[["model", "mean_balanced_accuracy"]].copy()
    hybrid_row = pd.DataFrame(
        [
            {
                "model": "Hybrid Quantum-Classical",
                "mean_balanced_accuracy": metadata["inner_cv_balanced_accuracy"],
            }
        ]
    )
    df = pd.concat([model_rows, hybrid_row], ignore_index=True)
    df = df.sort_values("mean_balanced_accuracy", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df["model"], df["mean_balanced_accuracy"], color="#54a24b")
    plt.xlabel("Mean Balanced Accuracy")
    plt.title("Model Comparison by Cross-Validated Balanced Accuracy")
    return save_figure("13_model_comparison_balanced_accuracy.png")


def plot_top_features(explainability: dict[str, object], top_k: int = 12) -> Path:
    rows = pd.DataFrame(explainability["feature_importance"][:top_k]).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(rows["feature"], rows["mean_absolute_attribution"], color="#e45756")
    plt.xlabel("Mean Absolute Attribution")
    plt.title("Top Global Features by Integrated Gradients")
    return save_figure("14_top_global_features.png")


def plot_signed_features(explainability: dict[str, object], top_k: int = 12) -> Path:
    rows = pd.DataFrame(explainability["feature_importance"][:top_k]).iloc[::-1]
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in rows["mean_signed_attribution"]]

    plt.figure(figsize=(8, 6))
    plt.barh(rows["feature"], rows["mean_signed_attribution"], color=colors)
    plt.xlabel("Mean Signed Attribution")
    plt.title("Top Feature Directionality")
    return save_figure("15_top_feature_directionality.png")


def plot_age_vs_probability(holdout: pd.DataFrame) -> Path:
    colors = holdout["actual_status"].map({"Dead": "#e45756", "Alive": "#4c78a8"})

    plt.figure(figsize=(7, 4.8))
    plt.scatter(holdout["age"], holdout["mortality_probability"], c=colors, alpha=0.8, edgecolors="black", linewidths=0.3)
    plt.xlabel("Age")
    plt.ylabel("Predicted Mortality Probability")
    plt.title("Age vs Predicted Mortality Probability")
    return save_figure("16_age_vs_probability.png")


def build_figure_index(figure_paths: list[Path], metadata: dict[str, object]) -> Path:
    captions = {
        "01_cohort_class_distribution.png": "Overall class balance of the aligned TCGA-GBM cohort used for mortality classification.",
        "02_train_holdout_distribution.png": "Class balance after the stratified outer split.",
        "03_age_distribution_by_status.png": "Age spread by outcome label in the aligned cohort.",
        "04_holdout_roc_curve.png": "ROC curve for the final hybrid classifier on the untouched holdout set.",
        "05_holdout_pr_curve.png": "Precision-recall curve for the final hybrid classifier on the holdout set.",
        "06_holdout_confusion_matrix.png": "Confusion matrix at the selected decision threshold.",
        "07_holdout_probability_distribution.png": "Distribution of predicted mortality probabilities by true class.",
        "08_holdout_calibration_curve.png": "Calibration profile of predicted mortality probabilities.",
        "09_threshold_sweep.png": "Effect of decision-threshold choice on balanced accuracy and F1.",
        "10_cv_auc_vs_balanced_accuracy.png": "Inner cross-validation candidate comparison across AUC and balanced accuracy.",
        "11_cv_overfit_gap.png": "Train-vs-validation AUC gap across candidate configurations.",
        "12_model_comparison_auc.png": "Cross-validated AUC comparison between the hybrid model and classical baselines.",
        "13_model_comparison_balanced_accuracy.png": "Cross-validated balanced accuracy comparison between the hybrid model and classical baselines.",
        "14_top_global_features.png": "Top global features ranked by mean absolute Integrated Gradients attribution.",
        "15_top_feature_directionality.png": "Signed feature attributions indicating whether features tend to increase or reduce mortality risk.",
        "16_age_vs_probability.png": "Relationship between patient age and model-predicted mortality probability on the holdout set.",
    }

    lines = [
        "# Thesis Figure Index",
        "",
        "## Key Results",
        f"- Holdout AUC: {metadata['holdout_metrics']['auc']:.4f}",
        f"- Holdout PR AUC: {metadata['holdout_metrics']['pr_auc']:.4f}",
        f"- Holdout balanced accuracy: {metadata['holdout_metrics']['balanced_accuracy']:.4f}",
        f"- Holdout F1: {metadata['holdout_metrics']['f1']:.4f}",
        f"- Holdout MCC: {metadata['holdout_metrics']['mcc']:.4f}",
        f"- Holdout Brier score: {metadata['holdout_metrics']['brier']:.4f}",
        "",
        "## Figures",
    ]

    for path in figure_paths:
        lines.append(f"- {path.name}: {captions.get(path.name, 'Thesis-ready figure.')}")

    index_path = OUTPUT_DIR / "figure_index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def main() -> None:
    metadata, holdout, cv_results, baseline, explainability = load_classifier_artifacts()
    dataset = load_aligned_dataset()
    y_true, y_score = build_curve_inputs(holdout)
    threshold = float(metadata["decision_threshold"])

    figure_paths = [
        plot_class_distribution(metadata),
        plot_split_distribution(metadata),
        plot_age_distribution(dataset),
        plot_roc_curve(y_true, y_score),
        plot_pr_curve(y_true, y_score),
        plot_confusion_matrix(holdout, threshold),
        plot_probability_histogram(holdout),
        plot_calibration_curve(y_true, y_score),
        plot_threshold_sweep(y_true, y_score),
        plot_cv_scatter(cv_results),
        plot_cv_overfit(cv_results),
        plot_baseline_comparison(baseline, metadata),
        plot_baseline_balanced_accuracy(baseline, metadata),
        plot_top_features(explainability),
        plot_signed_features(explainability),
        plot_age_vs_probability(holdout),
    ]
    index_path = build_figure_index(figure_paths, metadata)

    print(f"Saved {len(figure_paths)} figures to {OUTPUT_DIR}")
    print(f"Saved figure index to {index_path}")


if __name__ == "__main__":
    main()
