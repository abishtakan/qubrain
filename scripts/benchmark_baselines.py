from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qubrain.scripts.train_model import (
    ARTIFACT_DIR,
    INNER_CV_FOLDS,
    SEED,
    TEST_SIZE,
    build_selected_matrix,
    evaluate_predictions,
    fit_scaler,
    load_aligned_dataset,
    select_genes,
    set_seeds,
)


def create_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=SEED,
        ),
        "Classical MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=SEED,
        ),
    }


def main() -> None:
    set_seeds()
    dataset = load_aligned_dataset()
    indices = range(len(dataset.labels))
    train_idx, _ = train_test_split(
        list(indices),
        test_size=TEST_SIZE,
        stratify=dataset.labels,
        random_state=SEED,
    )

    age_train = dataset.age[train_idx]
    gender_train = dataset.gender[train_idx]
    genes_train = dataset.genes[train_idx]
    y_train = dataset.labels[train_idx]

    models = create_models()
    splitter = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        fold_metrics: list[dict[str, float | dict[str, int]]] = []
        for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(genes_train, y_train), start=1):
            print(f"{model_name}: fold {fold_index}/{INNER_CV_FOLDS}")
            selected_idx, _ = select_genes(
                train_genes=genes_train[fit_idx],
                train_y=y_train[fit_idx],
                gene_names=dataset.gene_names,
                n_top_genes=50,
            )
            X_fit = build_selected_matrix(age_train[fit_idx], gender_train[fit_idx], genes_train[fit_idx], selected_idx)
            X_val = build_selected_matrix(age_train[val_idx], gender_train[val_idx], genes_train[val_idx], selected_idx)

            scaler = fit_scaler(X_fit)
            X_fit_scaled = scaler.transform(X_fit)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_fit_scaled, y_train[fit_idx])
            probs = model.predict_proba(X_val_scaled)[:, 1]
            fold_metrics.append(evaluate_predictions(y_train[val_idx], probs, threshold=0.5))

        rows.append(
            {
                "model": model_name,
                "mean_auc": sum(float(metric["auc"]) for metric in fold_metrics) / len(fold_metrics),
                "mean_pr_auc": sum(float(metric["pr_auc"]) for metric in fold_metrics) / len(fold_metrics),
                "mean_accuracy": sum(float(metric["accuracy"]) for metric in fold_metrics) / len(fold_metrics),
                "mean_balanced_accuracy": sum(float(metric["balanced_accuracy"]) for metric in fold_metrics)
                / len(fold_metrics),
                "mean_f1": sum(float(metric["f1"]) for metric in fold_metrics) / len(fold_metrics),
                "mean_precision": sum(float(metric["precision"]) for metric in fold_metrics) / len(fold_metrics),
                "mean_recall": sum(float(metric["recall"]) for metric in fold_metrics) / len(fold_metrics),
                "mean_specificity": sum(float(metric["specificity"]) for metric in fold_metrics)
                / len(fold_metrics),
                "feature_count": 50,
                "protocol": "Outer-train-only 5-fold CV with fold-local feature selection and scaling",
            }
        )

    results = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    csv_path = ARTIFACT_DIR / "baseline_benchmark.csv"
    md_path = ARTIFACT_DIR / "baseline_benchmark.md"
    results.to_csv(csv_path, index=False)

    lines = [
        "# Classical Baseline Benchmark",
        "",
        "The baselines below were evaluated on the training partition only using the same fold-local feature-selection and scaling policy as the hybrid model.",
        "",
    ]
    for _, row in results.iterrows():
        lines.append(
            "- "
            f"{row['model']}: AUC={row['mean_auc']:.4f}, PR AUC={row['mean_pr_auc']:.4f}, "
            f"Balanced Acc={row['mean_balanced_accuracy']:.4f}, F1={row['mean_f1']:.4f}"
        )
    md_path.write_text("\n".join(lines))

    print(f"Saved classical benchmark CSV to {csv_path}")
    print(f"Saved classical benchmark report to {md_path}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
