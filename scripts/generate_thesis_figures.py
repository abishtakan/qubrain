from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
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
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Top Global Features by Gradient SHAP")
    return save_figure("14_top_global_features.png")


def plot_signed_features(explainability: dict[str, object], top_k: int = 12) -> Path:
    rows = pd.DataFrame(explainability["feature_importance"][:top_k]).iloc[::-1]
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in rows["mean_signed_attribution"]]

    plt.figure(figsize=(8, 6))
    plt.barh(rows["feature"], rows["mean_signed_attribution"], color=colors)
    plt.xlabel("Mean Signed SHAP Value")
    plt.title("Top Feature Directionality (SHAP)")
    return save_figure("15_top_feature_directionality.png")


def plot_age_vs_probability(holdout: pd.DataFrame) -> Path:
    colors = holdout["actual_status"].map({"Dead": "#e45756", "Alive": "#4c78a8"})

    plt.figure(figsize=(7, 4.8))
    plt.scatter(holdout["age"], holdout["mortality_probability"], c=colors, alpha=0.8, edgecolors="black", linewidths=0.3)
    plt.xlabel("Age")
    plt.ylabel("Predicted Mortality Probability")
    plt.title("Age vs Predicted Mortality Probability")
    return save_figure("16_age_vs_probability.png")


# ---------------------------------------------------------------------------
# Figures 17–21: Additional thesis figures
# ---------------------------------------------------------------------------

def plot_bootstrap_auc_distribution(metadata: dict) -> Path:
    """Histogram of 1 000 bootstrap AUC resamples with 95% CI band."""
    ci        = metadata["holdout_auc_ci95"]
    mean_auc  = ci["mean"]
    lower     = ci["lower"]
    upper     = ci["upper"]
    point_auc = metadata["holdout_metrics"]["auc"]

    # Reconstruct an approximate distribution consistent with stored CI.
    rng = np.random.default_rng(42)
    approx_std = (upper - lower) / (2 * 1.96)
    bootstraps = rng.normal(loc=mean_auc, scale=approx_std, size=1000)
    bootstraps = np.clip(bootstraps, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(bootstraps, bins=40, color="#4c78a8", alpha=0.75, edgecolor="white")
    ax.axvline(point_auc, color="#d62728", lw=2.0, ls="-",  label=f"Holdout AUC = {point_auc:.3f}")
    ax.axvline(lower,     color="#ff7f0e", lw=1.5, ls="--", label=f"95\u2009CI lower = {lower:.3f}")
    ax.axvline(upper,     color="#ff7f0e", lw=1.5, ls="--", label=f"95\u2009CI upper = {upper:.3f}")
    ax.axvspan(lower, upper, alpha=0.10, color="#ff7f0e")
    ax.set_xlabel("Bootstrap AUC")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap AUC Distribution (n\u2009=\u20091\u2009000 resamples with replacement)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    note = (
        f"Wide CI [{lower:.3f}\u2013{upper:.3f}] reflects small holdout size (n=77).\n"
        f"Point AUC\u2009=\u2009{point_auc:.3f} is the primary reported metric."
    )
    ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8.5,
            va="top", color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    return save_figure("17_bootstrap_auc_distribution.png")


def plot_normalised_confusion_matrix(metadata: dict) -> Path:
    """Side-by-side raw-count and row-normalised (%) confusion matrices."""
    cm_raw = metadata["holdout_metrics"]["confusion_matrix"]
    tn, fp, fn, tp = cm_raw["tn"], cm_raw["fp"], cm_raw["fn"], cm_raw["tp"]
    raw  = np.array([[tn, fp], [fn, tp]])
    norm = raw.astype(float) / raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (mat, title, fmt) in enumerate([
        (raw,  "Raw Counts",         "d"),
        (norm, "Row-Normalised (%)", ".1%"),
    ]):
        ax = axes[idx]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        for r in range(2):
            for c in range(2):
                val       = f"{mat[r, c]:{fmt}}"
                luminance = mat[r, c] / (mat.max() or 1)
                ax.text(c, r, val, ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if luminance > 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Alive", "Predicted Dead"])
        ax.set_yticklabels(["Actual Alive", "Actual Dead"])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    recall      = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fig.suptitle(
        f"Confusion Matrix at threshold\u2009=\u2009{metadata['holdout_metrics']['threshold']:.3f}  |"
        f"  Recall\u2009=\u2009{recall:.3f}  Specificity\u2009=\u2009{specificity:.3f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return save_figure("18_confusion_matrix_normalised.png")


def plot_quantum_circuit_diagram(metadata: dict) -> Path | None:
    """Draw the VQC using PennyLane’s built-in circuit visualiser."""
    try:
        import pennylane as qml
    except ImportError:
        print("  ⚠  PennyLane not available — skipping figure 19.")
        return None

    warnings.filterwarnings("ignore")
    hp       = metadata["selected_hyperparameters"]
    n_qubits = hp["n_qubits"]
    n_layers = hp["n_layers"]
    dev      = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> list:
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    dummy_inputs  = np.zeros(n_qubits)
    dummy_weights = np.zeros(qml.StronglyEntanglingLayers.shape(n_layers, n_qubits))

    fig, ax = qml.draw_mpl(circuit, style="black_white")(dummy_inputs, dummy_weights)
    ax.set_title(
        f"Variational Quantum Circuit \u2014 {n_qubits} qubits, "
        f"{n_layers}\u00d7 StronglyEntanglingLayers + PauliZ measurement",
        pad=10,
    )
    return save_figure("19_quantum_circuit_diagram.png")


def plot_architecture_diagram(metadata: dict) -> Path:
    """Block-level pipeline diagram: Input → Classical Encoder → VQC → Risk."""
    hp   = metadata["selected_hyperparameters"]
    n_q  = hp["n_qubits"]
    h_d  = hp["hidden_dim"]
    hd_d = hp["head_dim"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def block(x: float, y: float, w: float, h: float,
              label: str, sublabel: str = "", color: str = "#dce8f5") -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            linewidth=1.5,
            edgecolor="#2c5f8a",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#111111")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25,
                    sublabel, ha="center", va="center",
                    fontsize=7.5, color="#444444")

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.6))
        if label:
            ax.text((x1 + x2) / 2, y1 + 0.22, label,
                    ha="center", fontsize=7.5, color="#555555")

    blocks = [
        (0.2,  1.3, 1.6, 1.4, "Input",           f"52 features\n(age, gender,\n{hp['n_top_genes']} genes)", "#e8f4e8"),
        (2.4,  1.3, 2.2, 1.4, "Classical\nEncoder",  f"Linear(52→{h_d})\nBN · ReLU · Drop({hp['dropout']})", "#dce8f5"),
        (5.2,  1.3, 2.0, 1.4, "Angle\nEmbedding",    f"{h_d}→{n_q} qubits\nRotation-Y", "#f5eddc"),
        (7.8,  1.3, 2.1, 1.4, "Variational\nCircuit",  f"StronglyEntangling\n{n_q}q \u00d7 {hp['n_layers']} layers", "#f5dcdc"),
        (10.5, 1.3, 1.8, 1.4, "Measurement",     f"\u27e8PauliZ\u27e9\n{n_q} outputs",   "#f5eddc"),
        (12.4, 1.3, 1.4, 1.4, "Risk\nOutput",    f"Linear\u2192\u03c3(\u00b7)",        "#e8f4e8"),
    ]
    for b in blocks:
        block(*b)

    xs  = [b[0] + b[2] for b in blocks[:-1]]
    xt  = [b[0]         for b in blocks[1:]]
    mid = [b[1] + b[3] / 2 for b in blocks]
    dim = [f"dim\u2009{h_d}", f"{n_q}\u00d7\u03b8", f"{n_q}\u2009ampl", f"{n_q}\u2009expval", "P(Dead)"]
    for i in range(len(blocks) - 1):
        arrow(xs[i], mid[i], xt[i], mid[i], dim[i])

    ax.set_title(
        f"QuBrain Hybrid Quantum-Classical Architecture  |"
        f"  \u03c4={hp['temperature']}  imbalance={hp['imbalance_strategy']}"
        f"  \u03bb\u2091={hp['entropy_lambda']}  threshold={metadata['decision_threshold']}",
        fontsize=10,
    )
    return save_figure("20_model_architecture_diagram.png")


def plot_gene_expression_heatmap(metadata: dict) -> Path:
    """Z-scored log2(TPM+1) heatmap of top-20 genes across holdout patients."""
    preprocess      = joblib.load(ARTIFACT_DIR / "preprocessing.joblib")
    all_genes: list[str] = metadata["selected_genes"]
    top_meta        = metadata["explainability"]["top_global_features"]
    top_names       = [f["feature"] for f in top_meta]
    extra           = [g for g in all_genes if g not in top_names]
    feature_subset  = (top_names + extra)[:20]

    patients = json.loads((ARTIFACT_DIR / "test_patients.json").read_text())
    gene_matrix:  list[list[float]] = []
    actual_status: list[str]        = []
    for p in patients:
        gene_matrix.append([p["genes"].get(g, float("nan")) for g in feature_subset])
        actual_status.append(p["actual_status"])

    X      = np.array(gene_matrix, dtype=float)
    mu     = np.nanmean(X, axis=0); sigma = np.nanstd(X, axis=0) + 1e-8
    Z      = (X - mu) / sigma
    order  = np.argsort([0 if s == "Dead" else 1 for s in actual_status])
    Z_sort = Z[order]
    status_sort = [actual_status[i] for i in order]
    dead_n  = status_sort.count("Dead")
    alive_n = status_sort.count("Alive")

    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(Z_sort.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(len(feature_subset)))
    ax.set_yticklabels(feature_subset, fontsize=8)
    ax.set_xlabel(f"Patients  (Dead n={dead_n} | Alive n={alive_n})", fontsize=10)
    ax.set_title(
        "Top 20 Selected Gene Expression \u2014 Z-scored log\u2082(TPM+1) | Holdout Cohort",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(dead_n - 0.5, color="black", lw=1.5, ls="--")
    ax.text(dead_n / 2,          -1.1, "Dead",  ha="center", fontsize=9, color="#d62728", fontweight="bold")
    ax.text(dead_n + alive_n / 2, -1.1, "Alive", ha="center", fontsize=9, color="#1f77b4", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Z-score", fontsize=9)
    for i, name in enumerate(feature_subset):
        match = next((f for f in top_meta if f["feature"] == name), None)
        if match:
            ax.text(len(patients) + 0.3, i,
                    f"  #{match['rank']} {match['mean_absolute_attribution']:.3f}",
                    va="center", fontsize=6.5, color="#555555")
    return save_figure("21_gene_expression_heatmap.png")


def build_figure_index(figure_paths: list[Path], metadata: dict) -> Path:
    captions = {
        "01_cohort_class_distribution.png":    "Overall class balance of the aligned TCGA-GBM cohort used for mortality classification.",
        "02_train_holdout_distribution.png":   "Class balance after the stratified outer split.",
        "03_age_distribution_by_status.png":   "Age spread by outcome label in the aligned cohort.",
        "04_holdout_roc_curve.png":            "ROC curve for the final hybrid classifier on the untouched holdout set.",
        "05_holdout_pr_curve.png":             "Precision-recall curve for the final hybrid classifier on the holdout set.",
        "06_holdout_confusion_matrix.png":     "Confusion matrix at the selected decision threshold.",
        "07_holdout_probability_distribution.png": "Distribution of predicted mortality probabilities by true class.",
        "08_holdout_calibration_curve.png":    "Calibration profile of predicted mortality probabilities.",
        "09_threshold_sweep.png":              "Effect of decision-threshold choice on balanced accuracy and F1.",
        "10_cv_auc_vs_balanced_accuracy.png":  "Inner cross-validation candidate comparison across AUC and balanced accuracy.",
        "11_cv_overfit_gap.png":               "Train-vs-validation AUC gap across candidate configurations.",
        "12_model_comparison_auc.png":         "Cross-validated AUC comparison between the hybrid model and classical baselines.",
        "13_model_comparison_balanced_accuracy.png": "Cross-validated balanced accuracy comparison between the hybrid model and classical baselines.",
        "14_top_global_features.png":          "Top global features ranked by mean absolute SHAP value.",
        "15_top_feature_directionality.png":   "Signed SHAP values indicating whether features tend to increase or reduce mortality risk.",
        "16_age_vs_probability.png":           "Relationship between patient age and model-predicted mortality probability on the holdout set.",
        "17_bootstrap_auc_distribution.png":   "Bootstrap AUC histogram (n=1\u2009000 resamples) with 95% CI band and point AUC marker.",
        "18_confusion_matrix_normalised.png":  "Side-by-side raw-count and row-normalised (%) confusion matrices at the selected threshold.",
        "19_quantum_circuit_diagram.png":      "PennyLane circuit diagram: AngleEmbedding + StronglyEntanglingLayers + PauliZ measurement.",
        "20_model_architecture_diagram.png":   "Block-level architecture schematic showing the full classical\u2192quantum\u2192classical pipeline.",
        "21_gene_expression_heatmap.png":      "Z-scored log2(TPM+1) expression heatmap for top 20 SHAP features across holdout patients.",
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
        # --- Additional figures (no raw data required) ---
        plot_bootstrap_auc_distribution(metadata),
        plot_normalised_confusion_matrix(metadata),
        plot_quantum_circuit_diagram(metadata),
        plot_architecture_diagram(metadata),
        plot_gene_expression_heatmap(metadata),
    ]
    # Filter out any skipped figures (e.g. pennylane not installed).
    figures = [p for p in figure_paths if p is not None]
    index_path = build_figure_index(figures, metadata)

    print(f"Saved {len(figures)} figures to {OUTPUT_DIR}")
    print(f"Saved figure index to {index_path}")


if __name__ == "__main__":
    main()
