from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qubrain.backend.app.explainability import build_global_explainability
from qubrain.scripts.train_survival_model import (
    FINAL_ENSEMBLE_SEEDS,
    SURVIVAL_CATEGORICAL_COLUMNS,
    SurvivalDataset,
    fit_category_encoder,
    integrated_gradients,
    load_aligned_survival_dataset,
    transform_category_frame,
)

ARTIFACT_DIR = PROJECT_ROOT / "qubrain" / "backend" / "model_artifacts_survival_ranges_3band_ordinal"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
INNER_CV_FOLDS = 5
DEVICE = torch.device("cpu")
BAND_NAMES = ["lt_6", "m6_18", "gt_18"]
BAND_LABELS = {
    0: "< 6 months",
    1: "6-18 months",
    2: "> 18 months",
}
BAND_EDGES_MONTHS = [-np.inf, 6.0, 18.0, np.inf]


@dataclass(frozen=True)
class RangeModelConfig:
    n_top_genes: int
    n_qubits: int
    n_layers: int
    learning_rate: float
    dropout: float
    max_epochs: int
    patience: int
    hidden_dim: int = 64
    head_dim: int = 32
    weight_decay: float = 0.0
    label_smoothing: float = 0.05
    min_category_count: int = 6


@dataclass
class RangeDataset:
    age: np.ndarray
    gender: np.ndarray
    categorical_clinical: pd.DataFrame
    genes: np.ndarray
    gene_names: list[str]
    band_index: np.ndarray
    band_name: np.ndarray
    original_event: np.ndarray
    original_months: np.ndarray


@dataclass
class RangeTrainingOutcome:
    model: "HybridQuantumRangeClassifier"
    train_probs: np.ndarray
    val_probs: np.ndarray
    best_epoch: int
    train_metrics: dict[str, float | dict[str, int]]
    val_metrics: dict[str, float | dict[str, int]]


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_range_dataset(base_dataset: SurvivalDataset) -> RangeDataset:
    months = base_dataset.time_days / 30.44
    evaluable_mask = (base_dataset.event == 1) | ((base_dataset.event == 0) & (months > 18.0))
    band_index = pd.cut(
        months[evaluable_mask],
        bins=BAND_EDGES_MONTHS,
        labels=list(range(len(BAND_NAMES))),
    ).astype(int)
    band_name = np.array([BAND_NAMES[index] for index in band_index], dtype=object)
    return RangeDataset(
        age=base_dataset.age[evaluable_mask],
        gender=base_dataset.gender[evaluable_mask],
        categorical_clinical=base_dataset.categorical_clinical.iloc[evaluable_mask].reset_index(drop=True),
        genes=base_dataset.genes[evaluable_mask],
        gene_names=base_dataset.gene_names,
        band_index=np.asarray(band_index, dtype=int),
        band_name=band_name,
        original_event=base_dataset.event[evaluable_mask],
        original_months=months[evaluable_mask],
    )


def select_genes_for_ranges(
    train_genes: np.ndarray,
    train_band_index: np.ndarray,
    gene_names: list[str],
    n_top_genes: int,
) -> tuple[np.ndarray, list[str]]:
    non_constant_indices = np.where(np.var(train_genes, axis=0) > 0)[0]
    filtered_genes = train_genes[:, non_constant_indices]
    k = min(n_top_genes, filtered_genes.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(filtered_genes, train_band_index)
    indices = non_constant_indices[selector.get_support(indices=True)]
    names = [gene_names[index] for index in indices]
    return indices, names


def build_feature_matrix(
    age: np.ndarray,
    gender: np.ndarray,
    categorical_matrix: np.ndarray,
    genes: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    clinical = np.column_stack([age, gender]).astype(np.float32)
    selected_genes = genes[:, selected_idx].astype(np.float32)
    pieces = [clinical]
    if categorical_matrix.size:
        pieces.append(categorical_matrix.astype(np.float32))
    pieces.append(selected_genes)
    return np.hstack(pieces).astype(np.float32)


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(BAND_NAMES)).astype(np.float32)
    total = float(counts.sum())
    weights = np.where(counts > 0, total / (len(BAND_NAMES) * counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def encode_ordinal_levels(labels: np.ndarray) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int64)
    thresholds = np.arange(len(BAND_NAMES) - 1, dtype=np.int64)
    return (label_array[:, None] > thresholds[None, :]).astype(np.float32)


def ordinal_probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    class_probs = []
    class_probs.append(1.0 - cumulative[:, 0])
    for index in range(1, cumulative.shape[1]):
        class_probs.append(cumulative[:, index - 1] - cumulative[:, index])
    class_probs.append(cumulative[:, -1])
    return torch.stack(class_probs, dim=1)


def evaluate_range_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float | dict[str, int]]:
    preds = np.argmax(probs, axis=1)
    labels = np.arange(len(BAND_NAMES))
    cm = confusion_matrix(y_true, preds, labels=labels)
    quadratic_kappa = float(cohen_kappa_score(y_true, preds, weights="quadratic"))
    if np.isnan(quadratic_kappa):
        quadratic_kappa = 0.0
    band_error = np.abs(preds - y_true)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "quadratic_kappa": quadratic_kappa,
        "within_one_band_accuracy": float(np.mean(band_error <= 1)),
        "mean_absolute_band_error": float(np.mean(band_error)),
        "confusion_matrix": {
            BAND_NAMES[i]: {BAND_NAMES[j]: int(cm[i, j]) for j in range(len(BAND_NAMES))}
            for i in range(len(BAND_NAMES))
        },
    }


class HybridQuantumRangeClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_qubits: int,
        n_layers: int,
        hidden_dim: int,
        head_dim: int,
        dropout: float,
        n_classes: int = len(BAND_NAMES),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout = dropout
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),
        )
        self.q_layer = self._build_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.threshold_base = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        if n_classes > 2:
            self.threshold_deltas = nn.Parameter(torch.zeros(n_classes - 2, dtype=torch.float32))
        else:
            self.register_parameter("threshold_deltas", None)

    @staticmethod
    def _build_quantum_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch")
        def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(index)) for index in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def get_init_params(self) -> dict[str, int | float]:
        return {
            "n_features": self.n_features,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
        }

    def _ordered_thresholds(self) -> torch.Tensor:
        if self.threshold_deltas is None:
            return self.threshold_base
        deltas = torch.nn.functional.softplus(self.threshold_deltas)
        return torch.cat([self.threshold_base, self.threshold_base + torch.cumsum(deltas, dim=0)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        quantum_input = (encoded + 1) * (math.pi / 2)
        quantum_output = self.q_layer(quantum_input)
        score = self.head(quantum_output)
        thresholds = self._ordered_thresholds().view(1, -1)
        return score - thresholds


def instantiate_model(config: RangeModelConfig, n_features: int) -> HybridQuantumRangeClassifier:
    return HybridQuantumRangeClassifier(
        n_features=n_features,
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        head_dim=config.head_dim,
        dropout=config.dropout,
    )


class OrdinalRangeLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        thresholds = torch.arange(logits.shape[1], device=logits.device).view(1, -1)
        ordinal_targets = (labels.view(-1, 1) > thresholds).to(dtype=logits.dtype)
        sample_weights = self.class_weights[labels]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            ordinal_targets,
            reduction="none",
        )
        return (loss.mean(dim=1) * sample_weights).mean()


def infer_probabilities(model: HybridQuantumRangeClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE))
        return ordinal_probabilities_from_logits(logits).cpu().numpy()


def fit_range_model_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: RangeModelConfig,
) -> RangeTrainingOutcome:
    model = instantiate_model(config=config, n_features=X_train.shape[1]).to(DEVICE)
    class_weights = compute_class_weights(y_train)
    criterion = OrdinalRangeLoss(class_weights=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(DEVICE)

    best_score = -1.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_probs = infer_probabilities(model, X_val)
        val_metrics = evaluate_range_predictions(y_val, val_probs)
        val_score = float(val_metrics["macro_f1"])
        scheduler.step(val_score)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is None:  # pragma: no cover
        raise RuntimeError("Range training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    train_probs = infer_probabilities(model, X_train)
    val_probs = infer_probabilities(model, X_val)
    train_metrics = evaluate_range_predictions(y_train, train_probs)
    val_metrics = evaluate_range_predictions(y_val, val_probs)
    return RangeTrainingOutcome(
        model=model,
        train_probs=train_probs,
        val_probs=val_probs,
        best_epoch=best_epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


def fit_final_range_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RangeModelConfig,
    fixed_epochs: int,
) -> HybridQuantumRangeClassifier:
    model = instantiate_model(config=config, n_features=X_train.shape[1]).to(DEVICE)
    class_weights = compute_class_weights(y_train)
    criterion = OrdinalRangeLoss(class_weights=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(DEVICE)

    for _ in range(max(1, fixed_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    return model


def fit_final_range_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RangeModelConfig,
    fixed_epochs: int,
    seeds: list[int],
) -> list[HybridQuantumRangeClassifier]:
    models: list[HybridQuantumRangeClassifier] = []
    for seed in seeds:
        set_seeds(seed)
        model = fit_final_range_model(
            X_train=X_train,
            y_train=y_train,
            config=config,
            fixed_epochs=fixed_epochs,
        )
        models.append(model)
    return models


def infer_ensemble_probabilities(models: list[HybridQuantumRangeClassifier], X: np.ndarray) -> np.ndarray:
    probs = [infer_probabilities(model, X) for model in models]
    return np.mean(np.stack(probs, axis=0), axis=0)


def integrated_gradients_for_range_model(
    model: HybridQuantumRangeClassifier,
    inputs: np.ndarray,
    baseline: np.ndarray,
    steps: int = 24,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()

    input_array = np.asarray(inputs, dtype=np.float32)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    baseline_array = np.asarray(baseline, dtype=np.float32).reshape(-1)
    baseline_t = torch.from_numpy(baseline_array).to(device)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]
    risk_weights = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32, device=device)

    attributions: list[np.ndarray] = []
    for row in input_array:
        input_t = torch.from_numpy(row).to(device)
        total_grads = torch.zeros_like(input_t)
        delta = input_t - baseline_t

        for alpha in alphas:
            model.zero_grad(set_to_none=True)
            interpolated = (baseline_t + (alpha * delta)).unsqueeze(0).clone().detach().requires_grad_(True)
            logits = model(interpolated)
            probs = ordinal_probabilities_from_logits(logits)
            risk_score = torch.sum(probs * risk_weights.unsqueeze(0))
            risk_score.backward()
            total_grads += interpolated.grad.detach()[0]

        average_grads = total_grads / float(len(alphas))
        attribution = (delta * average_grads).detach().cpu().numpy().astype(np.float32)
        attributions.append(attribution)

    return np.vstack(attributions)


def integrated_gradients_ensemble(
    models: list[HybridQuantumRangeClassifier],
    inputs: np.ndarray,
    baseline: np.ndarray,
    steps: int = 24,
    device: str = "cpu",
) -> np.ndarray:
    all_attributions = [
        integrated_gradients_for_range_model(
            model=model,
            inputs=inputs,
            baseline=baseline,
            steps=steps,
            device=device,
        )
        for model in models
    ]
    return np.mean(np.stack(all_attributions, axis=0), axis=0)


def aggregate_fold_results(
    config: RangeModelConfig,
    fold_results: list[RangeTrainingOutcome],
) -> dict[str, float | int]:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "quadratic_kappa",
        "within_one_band_accuracy",
        "mean_absolute_band_error",
    ]
    summary = {
        **asdict(config),
        "mean_best_epoch": float(np.mean([result.best_epoch for result in fold_results])),
    }
    for metric_name in metric_names:
        train_values = [float(result.train_metrics[metric_name]) for result in fold_results]
        val_values = [float(result.val_metrics[metric_name]) for result in fold_results]
        summary[f"mean_train_{metric_name}"] = float(np.mean(train_values))
        summary[f"mean_val_{metric_name}"] = float(np.mean(val_values))

    summary["mean_overfit_gap_macro_f1"] = (
        float(summary["mean_train_macro_f1"]) - float(summary["mean_val_macro_f1"])
    )
    summary["mean_overfit_gap_balanced_accuracy"] = (
        float(summary["mean_train_balanced_accuracy"]) - float(summary["mean_val_balanced_accuracy"])
    )
    return summary


def rank_candidate_results(results: list[dict[str, float | int]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df.sort_values(
        by=[
            "mean_val_macro_f1",
            "mean_val_quadratic_kappa",
            "mean_val_balanced_accuracy",
            "mean_val_within_one_band_accuracy",
            "mean_overfit_gap_macro_f1",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def build_search_space(quick: bool) -> list[RangeModelConfig]:
    if quick:
        return [
            RangeModelConfig(
                n_top_genes=60,
                n_qubits=4,
                n_layers=2,
                learning_rate=0.001,
                dropout=0.2,
                max_epochs=18,
                patience=5,
            ),
            RangeModelConfig(
                n_top_genes=100,
                n_qubits=4,
                n_layers=2,
                learning_rate=0.0005,
                dropout=0.2,
                max_epochs=22,
                patience=6,
                hidden_dim=64,
                head_dim=32,
            ),
        ]

    return [
        RangeModelConfig(60, 4, 2, 0.0010, 0.25, 32, 8, min_category_count=8),
        RangeModelConfig(80, 4, 2, 0.0010, 0.20, 36, 9, hidden_dim=64, head_dim=32, min_category_count=6),
        RangeModelConfig(100, 4, 2, 0.0005, 0.20, 40, 10, hidden_dim=64, head_dim=32, min_category_count=6),
        RangeModelConfig(120, 4, 2, 0.0005, 0.15, 44, 10, hidden_dim=64, head_dim=32, min_category_count=6),
        RangeModelConfig(80, 6, 2, 0.0005, 0.15, 42, 10, hidden_dim=64, head_dim=32, min_category_count=6),
        RangeModelConfig(100, 6, 2, 0.0005, 0.15, 48, 12, hidden_dim=64, head_dim=32, min_category_count=6),
        RangeModelConfig(120, 6, 3, 0.0003, 0.10, 56, 12, hidden_dim=96, head_dim=48, min_category_count=5),
        RangeModelConfig(150, 6, 3, 0.0003, 0.10, 60, 14, hidden_dim=96, head_dim=48, min_category_count=5),
    ]


def run_inner_cv_search(
    age: np.ndarray,
    gender: np.ndarray,
    categorical_clinical: pd.DataFrame,
    genes: np.ndarray,
    band_index: np.ndarray,
    gene_names: list[str],
    quick: bool,
) -> tuple[RangeModelConfig, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=SEED)
    candidate_results: list[dict[str, float | int]] = []

    for config in build_search_space(quick=quick):
        print(
            "Evaluating range config "
            f"genes={config.n_top_genes}, qubits={config.n_qubits}, layers={config.n_layers}"
        )
        fold_outcomes: list[RangeTrainingOutcome] = []

        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(genes, band_index), start=1):
            print(f"  Fold {fold_index}/{INNER_CV_FOLDS}")
            selected_idx, _ = select_genes_for_ranges(
                train_genes=genes[train_idx],
                train_band_index=band_index[train_idx],
                gene_names=gene_names,
                n_top_genes=config.n_top_genes,
            )

            category_encoder = fit_category_encoder(
                categorical_clinical.iloc[train_idx].reset_index(drop=True),
                min_count=config.min_category_count,
            )
            train_categories = transform_category_frame(
                categorical_clinical.iloc[train_idx].reset_index(drop=True),
                category_encoder,
            )
            val_categories = transform_category_frame(
                categorical_clinical.iloc[val_idx].reset_index(drop=True),
                category_encoder,
            )

            X_train = build_feature_matrix(
                age[train_idx],
                gender[train_idx],
                train_categories,
                genes[train_idx],
                selected_idx,
            )
            X_val = build_feature_matrix(
                age[val_idx],
                gender[val_idx],
                val_categories,
                genes[val_idx],
                selected_idx,
            )

            scaler = fit_scaler(X_train)
            X_train_scaled = scaler.transform(X_train).astype(np.float32)
            X_val_scaled = scaler.transform(X_val).astype(np.float32)

            outcome = fit_range_model_with_validation(
                X_train=X_train_scaled,
                y_train=band_index[train_idx],
                X_val=X_val_scaled,
                y_val=band_index[val_idx],
                config=config,
            )
            fold_outcomes.append(outcome)
            print(
                "    "
                f"val_macro_f1={outcome.val_metrics['macro_f1']:.4f}, "
                f"val_bal_acc={outcome.val_metrics['balanced_accuracy']:.4f}, "
                f"epoch={outcome.best_epoch}"
            )

        candidate_results.append(aggregate_fold_results(config=config, fold_results=fold_outcomes))

    ranked = rank_candidate_results(candidate_results)
    best_row = ranked.iloc[0]
    best_config = RangeModelConfig(
        n_top_genes=int(best_row["n_top_genes"]),
        n_qubits=int(best_row["n_qubits"]),
        n_layers=int(best_row["n_layers"]),
        learning_rate=float(best_row["learning_rate"]),
        dropout=float(best_row["dropout"]),
        max_epochs=int(best_row["max_epochs"]),
        patience=int(best_row["patience"]),
        hidden_dim=int(best_row["hidden_dim"]),
        head_dim=int(best_row["head_dim"]),
        weight_decay=float(best_row["weight_decay"]),
        label_smoothing=float(best_row["label_smoothing"]),
        min_category_count=int(best_row["min_category_count"]),
    )
    return best_config, ranked


def summarize_dataset(
    base_dataset: SurvivalDataset,
    range_dataset: RangeDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, object]:
    train_bands = range_dataset.band_index[train_idx]
    test_bands = range_dataset.band_index[test_idx]
    evaluable_counts = pd.Series(range_dataset.band_name).value_counts().reindex(BAND_NAMES, fill_value=0)
    train_counts = pd.Series(train_bands).value_counts().reindex(range(len(BAND_NAMES)), fill_value=0)
    test_counts = pd.Series(test_bands).value_counts().reindex(range(len(BAND_NAMES)), fill_value=0)

    unresolved_censored = int(((base_dataset.event == 0) & ((base_dataset.time_days / 30.44) <= 18.0)).sum())
    retained_long_censored = int(((range_dataset.original_event == 0) & (range_dataset.original_months > 18.0)).sum())

    return {
        "source": "TCGA-GBM (NCI Genomic Data Commons)",
        "raw_expression_shape": {
            "samples": int(base_dataset.genes.shape[0]),
            "genes": int(base_dataset.genes.shape[1]),
        },
        "original_survival_cohort": {
            "samples": int(len(base_dataset.event)),
            "observed_events": int(base_dataset.event.sum()),
            "censored": int((base_dataset.event == 0).sum()),
        },
        "range_evaluable_cohort": {
            "samples": int(len(range_dataset.band_index)),
            "observed_events": int(range_dataset.original_event.sum()),
            "retained_long_followup_censored": retained_long_censored,
            "excluded_unresolved_censored": unresolved_censored,
            "band_distribution": {band: int(evaluable_counts[band]) for band in BAND_NAMES},
        },
        "split_summary": {
            "train_samples": int(len(train_idx)),
            "holdout_samples": int(len(test_idx)),
            "train_band_distribution": {BAND_NAMES[i]: int(train_counts[i]) for i in range(len(BAND_NAMES))},
            "holdout_band_distribution": {BAND_NAMES[i]: int(test_counts[i]) for i in range(len(BAND_NAMES))},
        },
    }


def write_markdown_report(metadata: dict[str, object], cv_results: pd.DataFrame, report_path: Path) -> None:
    dataset_summary = metadata["dataset_summary"]
    selected_hyperparameters = metadata["selected_hyperparameters"]
    holdout_metrics = metadata["holdout_metrics"]
    train_metrics = metadata["train_metrics"]

    lines = [
        "# Hybrid Quantum-Classical Survival Range Training Report",
        "",
        "## Task Definition",
        f"- Task: {metadata['task']}",
        f"- Target definition: {metadata['target_definition']}",
        "- Research framing: Clinically useful month-range prognosis prediction for prioritisation.",
        "",
        "## Dataset Summary",
        f"- Source: {dataset_summary['source']}",
        f"- Original survival cohort: {dataset_summary['original_survival_cohort']['samples']} samples",
        f"- Range-evaluable cohort: {dataset_summary['range_evaluable_cohort']['samples']} samples",
        f"- Excluded unresolved censored cases: {dataset_summary['range_evaluable_cohort']['excluded_unresolved_censored']}",
        "",
        "Band distribution:",
    ]

    for band in BAND_NAMES:
        lines.append(
            f"- {BAND_LABELS[BAND_NAMES.index(band)]}: "
            f"{dataset_summary['range_evaluable_cohort']['band_distribution'][band]}"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "- Stratified outer holdout split on month-range band labels",
            "- Inner 5-fold cross-validation for model selection",
            "- Fold-local range-aware gene ranking",
            "- Fold-local one-hot encoding of baseline categorical clinical covariates",
            "- Fold-local scaling",
            "- Hybrid quantum-classical ordinal prognosis-band training objective",
            "",
            "## Selected Hyperparameters",
            f"- Top genes: {selected_hyperparameters['n_top_genes']}",
            f"- Qubits: {selected_hyperparameters['n_qubits']}",
            f"- Layers: {selected_hyperparameters['n_layers']}",
            f"- Learning rate: {selected_hyperparameters['learning_rate']}",
            f"- Dropout: {selected_hyperparameters['dropout']}",
            f"- Final epochs: {metadata['final_training_epochs']}",
            f"- Ensemble size: {len(metadata['ensemble_seeds'])}",
            "",
            "## Cross-Validation Results",
            f"- Best inner-CV mean macro F1: {metadata['train_cv_macro_f1']:.4f}",
            f"- Best inner-CV mean balanced accuracy: {metadata['train_cv_balanced_accuracy']:.4f}",
            "Top candidates:",
        ]
    )

    for _, row in cv_results.head(5).iterrows():
        lines.append(
            "- "
            f"genes={int(row['n_top_genes'])}, qubits={int(row['n_qubits'])}, "
            f"layers={int(row['n_layers'])}, mean_val_macro_f1={row['mean_val_macro_f1']:.4f}, "
            f"mean_val_bal_acc={row['mean_val_balanced_accuracy']:.4f}, "
            f"mean_val_kappa={row['mean_val_quadratic_kappa']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Final Performance",
            f"- Train accuracy: {train_metrics['accuracy']:.4f}",
            f"- Train macro F1: {train_metrics['macro_f1']:.4f}",
            f"- Holdout accuracy: {holdout_metrics['accuracy']:.4f}",
            f"- Holdout balanced accuracy: {holdout_metrics['balanced_accuracy']:.4f}",
            f"- Holdout macro F1: {holdout_metrics['macro_f1']:.4f}",
            f"- Holdout weighted F1: {holdout_metrics['weighted_f1']:.4f}",
            f"- Holdout quadratic kappa: {holdout_metrics['quadratic_kappa']:.4f}",
            f"- Holdout within-one-band accuracy: {holdout_metrics['within_one_band_accuracy']:.4f}",
            f"- Holdout mean absolute band error: {holdout_metrics['mean_absolute_band_error']:.4f}",
            "",
            "## Interpretation Guardrail",
            "- This model predicts survival month ranges rather than exact death dates.",
            "- Patients censored before 18 months are excluded from range-label training and evaluation because their final band is unresolved.",
            "",
            "## Files",
            "- `metadata.json` contains the machine-readable result summary",
            "- `cv_results.csv` contains the prognosis-band hyperparameter search table",
            "- `holdout_predictions.csv` contains per-patient holdout month-range predictions",
            "- `explainability.json` contains global feature-importance rankings",
        ]
    )

    explainability = metadata.get("explainability", {})
    top_features = explainability.get("top_global_features", [])
    if top_features:
        lines.extend(
            [
                "",
                "## Explainability",
                "- Local explanations use Integrated Gradients against the median training-cohort baseline.",
                "- The attribution target is a severity-weighted short-survival score derived from class probabilities.",
                "",
                "Top global features:",
            ]
        )
        for row in top_features[:10]:
            lines.append(
                "- "
                f"{row['feature']}: mean_abs_attr={row['mean_absolute_attribution']:.4f}, "
                f"mean_signed_attr={row['mean_signed_attribution']:.4f}"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid quantum-classical survival range model.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller hyperparameter search for a faster smoke-test pass.",
    )
    args = parser.parse_args()

    set_seeds()
    print("Loading aligned survival dataset...")
    base_dataset = load_aligned_survival_dataset()
    dataset = build_range_dataset(base_dataset)

    indices = np.arange(len(dataset.band_index))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=dataset.band_index,
        random_state=SEED,
    )

    age_train, age_test = dataset.age[train_idx], dataset.age[test_idx]
    gender_train, gender_test = dataset.gender[train_idx], dataset.gender[test_idx]
    categorical_train = dataset.categorical_clinical.iloc[train_idx].reset_index(drop=True)
    categorical_test = dataset.categorical_clinical.iloc[test_idx].reset_index(drop=True)
    genes_train, genes_test = dataset.genes[train_idx], dataset.genes[test_idx]
    y_train, y_test = dataset.band_index[train_idx], dataset.band_index[test_idx]

    print("Running inner cross-validation hyperparameter search...")
    best_config, cv_results = run_inner_cv_search(
        age=age_train,
        gender=gender_train,
        categorical_clinical=categorical_train,
        genes=genes_train,
        band_index=y_train,
        gene_names=dataset.gene_names,
        quick=args.quick,
    )
    best_result = cv_results.iloc[0]

    selected_idx, selected_gene_names = select_genes_for_ranges(
        train_genes=genes_train,
        train_band_index=y_train,
        gene_names=dataset.gene_names,
        n_top_genes=best_config.n_top_genes,
    )
    category_encoder = fit_category_encoder(categorical_train, min_count=best_config.min_category_count)
    train_categories = transform_category_frame(categorical_train, category_encoder)
    test_categories = transform_category_frame(categorical_test, category_encoder)
    X_train_selected = build_feature_matrix(age_train, gender_train, train_categories, genes_train, selected_idx)
    X_test_selected = build_feature_matrix(age_test, gender_test, test_categories, genes_test, selected_idx)

    scaler = fit_scaler(X_train_selected)
    X_train_scaled = scaler.transform(X_train_selected).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_selected).astype(np.float32)

    final_epochs = int(best_config.max_epochs)
    print("Training final hybrid survival-range ensemble on the full training split...")
    final_models = fit_final_range_ensemble(
        X_train=X_train_scaled,
        y_train=y_train,
        config=best_config,
        fixed_epochs=final_epochs,
        seeds=FINAL_ENSEMBLE_SEEDS,
    )

    train_probs = infer_ensemble_probabilities(final_models, X_train_scaled)
    holdout_probs = infer_ensemble_probabilities(final_models, X_test_scaled)
    train_metrics = evaluate_range_predictions(y_train, train_probs)
    holdout_metrics = evaluate_range_predictions(y_test, holdout_probs)

    feature_order = ["age", "gender"] + category_encoder.feature_names + selected_gene_names
    reference_unscaled = np.median(X_train_selected, axis=0).astype(np.float32)
    reference_scaled = scaler.transform(reference_unscaled.reshape(1, -1)).astype(np.float32)[0]
    holdout_attributions = integrated_gradients_ensemble(
        models=final_models,
        inputs=X_test_scaled,
        baseline=reference_scaled,
        steps=24,
        device="cpu",
    )
    explainability = build_global_explainability(
        feature_names=feature_order,
        attributions=holdout_attributions,
    )

    model_path = ARTIFACT_DIR / "hybrid_survival_range_model_state.pt"
    preprocess_path = ARTIFACT_DIR / "preprocessing.joblib"
    metadata_path = ARTIFACT_DIR / "metadata.json"
    cv_results_path = ARTIFACT_DIR / "cv_results.csv"
    holdout_predictions_path = ARTIFACT_DIR / "holdout_predictions.csv"
    report_path = ARTIFACT_DIR / "research_report.md"
    explainability_path = ARTIFACT_DIR / "explainability.json"

    torch.save(
        {
            "ensemble": [
                {
                    "seed": seed,
                    "state_dict": model.state_dict(),
                    "model_params": model.get_init_params(),
                }
                for seed, model in zip(FINAL_ENSEMBLE_SEEDS, final_models)
            ],
            "n_features": int(X_train_scaled.shape[1]),
            "band_labels": BAND_LABELS,
        },
        model_path,
    )
    joblib.dump(
        {
            "scaler": scaler,
            "selected_genes": selected_gene_names,
            "category_encoder": category_encoder.categories,
            "feature_order": feature_order,
            "reference_unscaled": reference_unscaled.tolist(),
            "reference_scaled": reference_scaled.tolist(),
            "band_labels": BAND_LABELS,
            "band_edges_months": BAND_EDGES_MONTHS,
        },
        preprocess_path,
    )

    holdout_preds = np.argmax(holdout_probs, axis=1)
    holdout_predictions = pd.DataFrame(
        {
            "patient_index": np.arange(len(y_test)),
            "event_observed": dataset.original_event[test_idx],
            "observed_survival_months": dataset.original_months[test_idx],
            "true_band_index": y_test,
            "true_band": [BAND_LABELS[index] for index in y_test],
            "predicted_band_index": holdout_preds,
            "predicted_band": [BAND_LABELS[index] for index in holdout_preds],
            "age": age_test,
            "gender": np.where(gender_test == 1, "male", "female"),
        }
    )
    for class_index, band_name in enumerate(BAND_NAMES):
        holdout_predictions[f"prob_{band_name}"] = holdout_probs[:, class_index]
    holdout_predictions.to_csv(holdout_predictions_path, index=False)
    cv_results.to_csv(cv_results_path, index=False)

    metadata = {
        "task": "GBM survival month-range prediction",
        "target_definition": "Predict clinically useful prognosis bands: <6, 6-18, or >18 months",
        "research_positioning": "Hybrid quantum-classical ordinal 3-band prognosis modeling on TCGA-GBM for patient prioritisation.",
        "selected_model": "hybrid_quantum_classical_survival_range_3band_ordinal_ensemble",
        "band_definition": BAND_LABELS,
        "dataset_summary": summarize_dataset(base_dataset, dataset, train_idx=train_idx, test_idx=test_idx),
        "preprocessing": {
            "expression_transform": "log2(TPM + 1)",
            "clinical_features": ["age", "gender"] + SURVIVAL_CATEGORICAL_COLUMNS,
            "feature_selection": "ANOVA range-association ranking fitted on training data only",
            "scaling": "MinMaxScaler fitted on training data only",
            "categorical_encoding": "Training-fold one-hot encoding with rare-category collapse",
            "censoring_policy": "Deaths are band-labeled from observed survival; censored cases are retained only when follow-up exceeds 18 months.",
        },
        "validation_protocol": {
            "outer_split": "Stratified train/test split (80/20) on prognosis-band labels",
            "inner_cv": f"Stratified {INNER_CV_FOLDS}-fold cross-validation on the training partition",
            "selection_metric": "mean validation macro F1",
            "secondary_metrics": [
                "balanced_accuracy",
                "quadratic_kappa",
                "within_one_band_accuracy",
            ],
        },
        "selected_hyperparameters": asdict(best_config),
        "train_cv_macro_f1": float(best_result["mean_val_macro_f1"]),
        "train_cv_balanced_accuracy": float(best_result["mean_val_balanced_accuracy"]),
        "inner_cv_results_top5": cv_results.head(5).to_dict(orient="records"),
        "final_training_epochs": final_epochs,
        "ensemble_seeds": FINAL_ENSEMBLE_SEEDS,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "selected_genes": selected_gene_names,
        "feature_order": feature_order,
        "explainability": {
            "local_method": "Integrated Gradients relative to the median training-cohort feature profile",
            "global_method": "Mean absolute Integrated Gradients attribution on the outer holdout cohort",
            "attribution_target": "Severity-weighted short-survival score derived from prognosis-band probabilities",
            "baseline_reference": "Median training-cohort feature vector",
            "top_global_features": explainability["feature_importance"][:10],
        },
        "artifact_files": {
            "model": model_path.name,
            "preprocessing": preprocess_path.name,
            "metadata": metadata_path.name,
            "cv_results": cv_results_path.name,
            "holdout_predictions": holdout_predictions_path.name,
            "research_report": report_path.name,
            "explainability": explainability_path.name,
        },
        "notes": [
            "This model is evaluated on 3-band month-range correctness rather than exact survival-day prediction.",
            "Patients censored before 18 months are excluded because their final prognosis band is unresolved.",
            "Final reported probabilities are the mean output of multiple independently trained ordinal hybrid models.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    explainability_path.write_text(json.dumps(explainability, indent=2), encoding="utf-8")
    write_markdown_report(metadata=metadata, cv_results=cv_results, report_path=report_path)

    print(f"Saved survival-range model state to {model_path}")
    print(f"Saved preprocessing to {preprocess_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved CV results to {cv_results_path}")
    print(f"Saved holdout predictions to {holdout_predictions_path}")
    print(f"Saved research report to {report_path}")
    print(f"Saved explainability report to {explainability_path}")
    print("Selected configuration:")
    print(json.dumps(asdict(best_config), indent=2))
    print("Holdout metrics:")
    for key, value in holdout_metrics.items():
        if isinstance(value, dict):
            continue
        print(f"  {key}: {float(value):.4f}")


if __name__ == "__main__":
    main()
