"""
train_model.py — Research-grade training pipeline for the QuBrain hybrid
quantum-classical GBM mortality-status classifier.

Pipeline overview
-----------------
1. Load and align the TCGA-GBM RNA-seq expression matrix with clinical labels.
2. Create a stratified outer 80/20 holdout split (the holdout set is never used
   for any tuning or selection decision).
3. Run an inner 5-fold stratified cross-validation hyperparameter search over
   the predefined search space. Feature selection, scaling, and class balancing
   are all fitted inside each training fold to prevent information leakage.
4. Select the best configuration by mean validation ROC-AUC.
5. Retrain the selected configuration on the full training split for a fixed
   number of epochs derived from the cross-validation.
6. Evaluate once on the untouched holdout set and compute bootstrap CIs.
7. Compute global Integrated Gradients explainability on holdout samples.
8. Save all artifacts (model, preprocessing, metadata, CV results,
   holdout predictions, explainability, and the markdown research report).

Usage
-----
    python scripts/train_model.py          # full hyperparameter search
    python scripts/train_model.py --quick  # reduced search for smoke testing

Artifacts written
-----------------
    backend/model_artifacts/hybrid_model_state.pt
    backend/model_artifacts/preprocessing.joblib
    backend/model_artifacts/metadata.json
    backend/model_artifacts/cv_results.csv
    backend/model_artifacts/holdout_predictions.csv
    backend/model_artifacts/test_patients.json
    backend/model_artifacts/explainability.json
    backend/model_artifacts/research_report.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qubrain.backend.app.hybrid_model import HybridQuantumClassifier
from qubrain.backend.app.explainability import build_global_explainability, shap_explain

DATA_DIR = PROJECT_ROOT / "qubrain" / "data"
ARTIFACT_DIR = PROJECT_ROOT / "qubrain" / "backend" / "model_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
INNER_CV_FOLDS = 5
FINAL_BOOTSTRAP_SAMPLES = 1000
DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable hyperparameter configuration for one candidate model.

    Each field corresponds to a tuneable dimension in the search space.
    The dataclass is frozen so that instances can be hashed and used as
    dictionary keys in aggregation steps.
    """
    n_top_genes: int
    n_qubits: int
    n_layers: int
    learning_rate: float
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    entropy_lambda: float
    imbalance_strategy: str
    hidden_dim: int = 32
    head_dim: int = 16
    temperature: float = 0.5
    weight_decay: float = 1e-4


@dataclass
class Dataset:
    """Container for the fully aligned, label-annotated TCGA-GBM dataset."""
    age: np.ndarray
    gender: np.ndarray
    genes: np.ndarray
    gene_names: list[str]
    labels: np.ndarray


@dataclass
class TrainingOutcome:
    """Results produced by a single fold's training run, stored for aggregation."""
    model: HybridQuantumClassifier
    train_probs: np.ndarray
    val_probs: np.ndarray
    best_epoch: int
    best_threshold: float
    train_metrics: dict[str, float | dict[str, int]]
    val_metrics: dict[str, float | dict[str, int]]


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_single_gene_file(filepath: Path) -> pd.Series | None:
    """
    Parse one STAR gene-count TSV file and return a Series of protein-coding
    TPM values indexed by gene name.

    Returns None if the file is missing required columns or cannot be parsed.
    """
    try:
        df = pd.read_csv(filepath, sep="\t", skiprows=1, low_memory=False)
        if "gene_name" not in df.columns or "tpm_unstranded" not in df.columns:
            return None
        if "gene_type" in df.columns:
            df = df[df["gene_type"] == "protein_coding"]
        return df[["gene_name", "tpm_unstranded"]].dropna().set_index("gene_name")["tpm_unstranded"]
    except Exception as exc:
        print(f"Failed to read {filepath}: {exc}")
        return None


def load_expression_matrix() -> pd.DataFrame:
    """
    Discover and load all STAR gene-count files under ``data/gene_expression/``.

    Each file contributes one row (sample) to the expression matrix. Only
    protein-coding genes are retained and expression is log2-transformed:
    ``log2(TPM + 1)``.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, n_genes), rows indexed by GDC file ID.
    """
    gene_dir = DATA_DIR / "gene_expression"
    gene_files = sorted(gene_dir.rglob("*star_gene_counts*.tsv"))
    if not gene_files:
        raise FileNotFoundError(f"No STAR gene count files found in {gene_dir}")

    samples: dict[str, pd.Series] = {}
    for filepath in gene_files:
        file_id = filepath.parent.name
        series = _load_single_gene_file(filepath)
        if series is not None:
            samples[file_id] = series

    # Log-transform expression values.
    matrix = pd.DataFrame(samples).dropna(how="all")
    return np.log2(matrix.T + 1)


def load_clinical_data() -> pd.DataFrame:
    """
    Load the TCGA-GBM clinical TSV and build binary mortality labels.

    The target variable is derived from ``demographic.vital_status``:
    ``1 = Dead``, ``0 = Alive``. Missing age and gender values are imputed
    with the median and mode respectively.

    Returns
    -------
    pd.DataFrame
        Columns: case_id, age, gender (0/1), target (0/1).
    """
    clinical_file = DATA_DIR / "clinical.project-tcga-gbm.2026-01-08" / "clinical.tsv"
    df = pd.read_csv(clinical_file, sep="\t", low_memory=False)

    columns = {
        "cases.case_id": "case_id",
        "demographic.age_at_index": "age",
        "demographic.gender": "gender",
        "demographic.vital_status": "vital_status",
    }
    df = df[list(columns.keys())].rename(columns=columns)
    df = df.drop_duplicates(subset=["case_id"])
    df = df[df["vital_status"].isin(["Alive", "Dead"])].copy()

    # Build binary mortality labels.
    df["target"] = (df["vital_status"] == "Dead").astype(int)
    df["gender"] = df["gender"].str.lower().map({"male": 1, "female": 0})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())
    df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0]).astype(int)
    return df[["case_id", "age", "gender", "target"]]


def load_aligned_dataset() -> Dataset:
    """
    Join the expression matrix and clinical data via the GDC file-to-case ID
    mapping and return a single aligned ``Dataset``.

    Only samples that have both valid expression data AND a binary vital-status
    label are retained. Duplicate file IDs are dropped.
    """
    expression = load_expression_matrix()
    clinical = load_clinical_data()

    mapping_file = DATA_DIR / "file_case_mapping.csv"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_file}")

    mapping = pd.read_csv(mapping_file)
    aligned = clinical.merge(mapping, on="case_id", how="inner")
    # Keep only aligned clinical and expression records.
    aligned = aligned[aligned["file_id"].isin(expression.index)].copy()
    aligned = aligned.drop_duplicates(subset=["file_id"]).reset_index(drop=True)

    expression_aligned = expression.loc[aligned["file_id"]]
    return Dataset(
        age=aligned["age"].to_numpy(dtype=float),
        gender=aligned["gender"].to_numpy(dtype=int),
        genes=expression_aligned.to_numpy(dtype=float),
        gene_names=expression_aligned.columns.tolist(),
        labels=aligned["target"].to_numpy(dtype=int),
    )


def select_genes(
    train_genes: np.ndarray,
    train_y: np.ndarray,
    gene_names: list[str],
    n_top_genes: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Select the top ``n_top_genes`` genes from the training partition using
    univariate ANOVA F-statistic (``SelectKBest(f_classif)``).

    Zero-variance genes are filtered out first to avoid division-by-zero in
    the F-test. Feature selection is fitted on training data only to prevent
    leakage into the validation or test fold.

    Returns
    -------
    tuple
        ``(selected_indices, selected_gene_names)`` — integer column indices
        into the full gene matrix and the corresponding gene name strings.
    """
    # Fit feature ranking on training data only.
    non_constant_indices = np.where(np.var(train_genes, axis=0) > 0)[0]
    filtered_genes = train_genes[:, non_constant_indices]
    k = min(n_top_genes, filtered_genes.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(filtered_genes, train_y)
    indices = non_constant_indices[selector.get_support(indices=True)]
    names = [gene_names[index] for index in indices]
    return indices, names


def build_selected_matrix(
    age: np.ndarray,
    gender: np.ndarray,
    genes: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    """
    Assemble the final feature matrix ``[age, gender, gene_1, ..., gene_k]``
    by concatenating clinical covariates with the selected gene columns.

    Returns a float32 array of shape (n_samples, 2 + len(selected_idx)).
    """
    # Combine clinical covariates with selected genes.
    clinical = np.column_stack([age, gender])
    selected_genes = genes[:, selected_idx]
    return np.hstack([clinical, selected_genes]).astype(np.float32)


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    """Fit a MinMaxScaler on ``X_train`` and return it (not yet applied)."""
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def threshold_grid() -> np.ndarray:
    # Candidate decision thresholds.
    return np.linspace(0.1, 0.9, 81)


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Select the decision threshold that maximises balanced accuracy on the
    provided ground-truth labels and predicted probabilities.

    Searches over 81 evenly spaced candidate values in ``[0.1, 0.9]``.
    Using balanced accuracy as the selection metric is appropriate for
    imbalanced classes because it weights sensitivity and specificity equally.
    """
    best_threshold = 0.5
    best_score = -1.0
    for threshold in threshold_grid():
        preds = (probs >= threshold).astype(int)
        # Select threshold by balanced accuracy.
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float | dict[str, int]]:
    """
    Compute the full suite of classification metrics at a given threshold.

    Metrics returned: ROC-AUC, PR-AUC, accuracy, balanced accuracy, F1,
    precision, recall, specificity, MCC, Brier score, and the full
    confusion matrix (TN, FP, FN, TP).

    AUC is threshold-independent; all other metrics use ``threshold`` to
    binarise the probability scores.
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "brier": float(brier_score_loss(y_true, probs)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = FINAL_BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    """
    Estimate a 95% bootstrap confidence interval for holdout ROC-AUC.

    Resamples the holdout set ``n_bootstrap`` times with replacement and
    computes AUC on each resample. The 2.5th and 97.5th percentiles of the
    resulting distribution form the lower and upper CI bounds.

    Bootstraps that yield only one class in the resample are skipped to
    avoid AUC being undefined.

    Returns
    -------
    dict
        ``{"mean": ..., "lower": ..., "upper": ...}`` — all float.
    """
    # Estimate holdout AUC confidence interval.
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    sample_indices = np.arange(len(y_true))
    for _ in range(n_bootstrap):
        chosen = rng.choice(sample_indices, size=len(sample_indices), replace=True)
        y_boot = y_true[chosen]
        if np.unique(y_boot).size < 2:
            continue
        aucs.append(float(roc_auc_score(y_boot, probs[chosen])))

    if not aucs:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}

    return {
        "mean": float(np.mean(aucs)),
        "lower": float(np.quantile(aucs, 0.025)),
        "upper": float(np.quantile(aucs, 0.975)),
    }


def compute_risk_band_cutoffs(probs: np.ndarray, default_threshold: float) -> dict[str, float]:
    """
    Derive risk-band probability cutoffs from training score quantiles.

    The low/moderate/high bands are defined by the 33rd and 67th percentiles
    of the training set predicted probabilities. If these quantiles are
    degenerate (high_lower <= low_upper), symmetric offsets around the
    decision threshold are used as a fallback.
    """
    # Derive risk bands from training-score quantiles.
    low_upper = float(np.quantile(probs, 0.33))
    high_lower = float(np.quantile(probs, 0.67))

    if high_lower <= low_upper:
        low_upper = max(0.0, default_threshold - 0.08)
        high_lower = min(1.0, default_threshold + 0.08)

    return {
        "low_upper": low_upper,
        "high_lower": high_lower,
    }


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute inverse-frequency class weights for an imbalanced binary target.

    The weight for each class c is: ``total / (2 * count_c)``. This gives
    equal total loss contribution to both classes regardless of imbalance ratio.
    """
    # Compute class weights for imbalanced labels.
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    total = float(counts.sum())
    return {
        0: total / (2.0 * counts[0]) if counts[0] else 1.0,
        1: total / (2.0 * counts[1]) if counts[1] else 1.0,
    }


def apply_imbalance_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, float] | None]:
    """
    Apply the selected class-imbalance handling strategy to the training data.

    Must be called ONLY on the training partition (never on validation or test
    data) to prevent information leakage.

    Parameters
    ----------
    strategy : str
        One of:
        - ``"smote"``        — oversample the minority class with SMOTE.
        - ``"class_weight"`` — compute per-sample weights for the loss function.
        - ``"none"``         — no balancing.

    Returns
    -------
    tuple
        ``(X_fit, y_fit, class_weights)`` — X_fit and y_fit are the (possibly
        resampled) training arrays; class_weights is a dict or None.
    """
    if strategy == "smote":
        # Apply SMOTE on the training partition only.
        minority_count = int(np.bincount(y_train.astype(int), minlength=2).min())
        k_neighbors = min(5, max(1, minority_count - 1))
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled.astype(np.float32), y_resampled.astype(np.int64), None

    if strategy == "class_weight":
        return X_train, y_train, compute_class_weights(y_train)

    if strategy == "none":
        return X_train, y_train, None

    raise ValueError(f"Unsupported imbalance strategy: {strategy}")


class EntropyRegularizedBCELoss(nn.Module):
    """
    Binary cross-entropy loss with optional per-class weighting and entropy
    regularisation.

    Class weighting addresses the 81/19 class imbalance by scaling each
    sample's loss by its inverse-frequency weight.

    Entropy regularisation adds ``lambda_entropy * H(p)`` to the loss, which
    penalises over-confident predictions and acts as a form of soft calibration
    to prevent the model from collapsing to a trivial majority-class output.
    Setting ``lambda_entropy = 0`` disables this term.
    """
    def __init__(self, class_weights: dict[int, float] | None, lambda_entropy: float) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.lambda_entropy = lambda_entropy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the (optionally weighted, optionally entropy-regularised)
        binary cross-entropy for a batch of predictions.
        """
        epsilon = 1e-8
        p = torch.clamp(pred, epsilon, 1 - epsilon)
        # BCE with optional entropy regularization.
        per_sample_loss = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))

        if self.class_weights is not None:
            neg_weight = self.class_weights[0]
            pos_weight = self.class_weights[1]
            sample_weights = torch.where(target > 0.5, pos_weight, neg_weight)
            per_sample_loss = per_sample_loss * sample_weights

        loss = torch.mean(per_sample_loss)
        if self.lambda_entropy > 0:
            entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
            loss = loss + (self.lambda_entropy * torch.mean(entropy))
        return loss


def instantiate_model(config: ModelConfig, n_features: int) -> HybridQuantumClassifier:
    """Build a HybridQuantumClassifier from a ModelConfig and feature count."""
    return HybridQuantumClassifier(
        n_features=n_features,
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        head_dim=config.head_dim,
        dropout=config.dropout,
        temperature=config.temperature,
    )


def infer_probabilities(model: HybridQuantumClassifier, X: np.ndarray) -> np.ndarray:
    """Run the model in eval mode and return positive-class probabilities as a NumPy array."""
    model.eval()
    with torch.no_grad():
        # Return positive-class probabilities.
        return model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()


def fit_hybrid_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
    seed: int,
) -> TrainingOutcome:
    """
    Train a HybridQuantumClassifier on one fold with early stopping.

    The best model state (by validation AUC) is checkpointed and restored at
    the end of training. The decision threshold is also selected on the
    validation fold using balanced accuracy as the criterion.

    Class imbalance handling and the custom entropy-regularised BCE loss are
    applied only to the training partition — the validation set is never
    modified or balanced.

    Returns
    -------
    TrainingOutcome
        Contains the best model, predicted probabilities, best epoch and
        threshold, and train/val metric dicts.
    """
    # Apply imbalance handling to fit data only.
    X_fit, y_fit, class_weights = apply_imbalance_strategy(X_train, y_train, config.imbalance_strategy, seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit.astype(np.float32)),
            torch.from_numpy(y_fit.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = instantiate_model(config=config, n_features=X_train.shape[1]).to(DEVICE)
    criterion = EntropyRegularizedBCELoss(class_weights=class_weights, lambda_entropy=config.entropy_lambda)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(DEVICE)
    best_auc = -1.0
    best_epoch = 1
    best_threshold = 0.5
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).cpu().numpy()

        # Track the best validation AUC state.
        val_auc = float(roc_auc_score(y_val, val_probs))
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_threshold = choose_threshold(y_val, val_probs)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is None: 
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    train_probs = infer_probabilities(model, X_train.astype(np.float32))
    val_probs = infer_probabilities(model, X_val.astype(np.float32))
    train_metrics = evaluate_predictions(y_train, train_probs, best_threshold)
    val_metrics = evaluate_predictions(y_val, val_probs, best_threshold)

    return TrainingOutcome(
        model=model,
        train_probs=train_probs,
        val_probs=val_probs,
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


def fit_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ModelConfig,
    fixed_epochs: int,
    seed: int,
) -> HybridQuantumClassifier:
    """
    Retrain the best configuration on the full training split.

    Unlike ``fit_hybrid_with_validation``, there is no validation set or
    early stopping here. The number of epochs is fixed to the mean best epoch
    observed across the inner cross-validation folds, avoiding the need to
    hold out any data from the final training set.
    """
    # Retrain the selected configuration on the full training split.
    X_fit, y_fit, class_weights = apply_imbalance_strategy(X_train, y_train, config.imbalance_strategy, seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit.astype(np.float32)),
            torch.from_numpy(y_fit.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = instantiate_model(config=config, n_features=X_train.shape[1]).to(DEVICE)
    criterion = EntropyRegularizedBCELoss(class_weights=class_weights, lambda_entropy=config.entropy_lambda)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for _ in range(max(1, fixed_epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def aggregate_fold_results(
    config: ModelConfig,
    fold_results: list[TrainingOutcome],
) -> dict[str, float | str | int]:
    """
    Average per-fold metrics for a single candidate configuration.

    Returns a flat dict merging the config hyperparameters with mean
    train/val AUC, balanced accuracy, F1, specificity, Brier score, best
    epoch, best threshold, and the overfitting gap (train minus val AUC).
    This dict forms one row of the CV results table.
    """
    # Aggregate fold metrics for model selection.
    mean_train_auc = float(np.mean([result.train_metrics["auc"] for result in fold_results]))
    mean_val_auc = float(np.mean([result.val_metrics["auc"] for result in fold_results]))
    mean_val_balanced_accuracy = float(
        np.mean([result.val_metrics["balanced_accuracy"] for result in fold_results])
    )
    mean_val_f1 = float(np.mean([result.val_metrics["f1"] for result in fold_results]))
    mean_val_specificity = float(np.mean([result.val_metrics["specificity"] for result in fold_results]))
    mean_val_brier = float(np.mean([result.val_metrics["brier"] for result in fold_results]))
    mean_best_epoch = float(np.mean([result.best_epoch for result in fold_results]))
    mean_best_threshold = float(np.mean([result.best_threshold for result in fold_results]))
    return {
        **asdict(config),
        "mean_train_auc": mean_train_auc,
        "mean_val_auc": mean_val_auc,
        "mean_val_balanced_accuracy": mean_val_balanced_accuracy,
        "mean_val_f1": mean_val_f1,
        "mean_val_specificity": mean_val_specificity,
        "mean_val_brier": mean_val_brier,
        "mean_overfit_gap_auc": mean_train_auc - mean_val_auc,
        "mean_best_epoch": mean_best_epoch,
        "mean_best_threshold": mean_best_threshold,
    }


def rank_candidate_results(results: list[dict[str, float | str | int]]) -> pd.DataFrame:
    """
    Sort candidate configurations by mean validation AUC (descending),
    then balanced accuracy (descending), then Brier score (ascending).

    Returns a DataFrame where row 0 is the best configuration.
    """
    df = pd.DataFrame(results)
    return df.sort_values(
        by=["mean_val_auc", "mean_val_balanced_accuracy", "mean_val_brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_search_space(quick: bool) -> list[ModelConfig]:
    """
    Return the list of candidate ``ModelConfig`` objects to evaluate.

    When ``quick=True`` a reduced set of 5 configurations is returned for
    fast smoke-testing. The full search space contains 8 configurations
    covering different qubit counts, layer depths, learning rates, dropout
    values, entropy regularisation strengths, and imbalance strategies.
    """
    if quick:
        # Reduced search space for smoke tests.
        return [
            ModelConfig(
                n_top_genes=50,
                n_qubits=4,
                n_layers=2,
                learning_rate=0.001,
                dropout=0.2,
                batch_size=32,
                max_epochs=30,
                patience=6,
                entropy_lambda=0.05,
                imbalance_strategy="smote",
            ),
            ModelConfig(
                n_top_genes=50,
                n_qubits=6,
                n_layers=2,
                learning_rate=0.001,
                dropout=0.2,
                batch_size=32,
                max_epochs=30,
                patience=6,
                entropy_lambda=0.05,
                imbalance_strategy="class_weight",
            ),
            ModelConfig(
                n_top_genes=50,
                n_qubits=6,
                n_layers=2,
                learning_rate=0.001,
                dropout=0.1,
                batch_size=32,
                max_epochs=60,
                patience=10,
                entropy_lambda=0.0,
                imbalance_strategy="smote",
                hidden_dim=64,
                head_dim=32,
                temperature=0.5,
                weight_decay=0.0,
            ),
            ModelConfig(
                n_top_genes=50,
                n_qubits=6,
                n_layers=3,
                learning_rate=0.001,
                dropout=0.1,
                batch_size=32,
                max_epochs=70,
                patience=12,
                entropy_lambda=0.02,
                imbalance_strategy="smote",
                hidden_dim=64,
                head_dim=32,
                temperature=0.35,
                weight_decay=0.0,
            ),
            ModelConfig(
                n_top_genes=50,
                n_qubits=6,
                n_layers=2,
                learning_rate=0.0005,
                dropout=0.1,
                batch_size=32,
                max_epochs=80,
                patience=12,
                entropy_lambda=0.0,
                imbalance_strategy="class_weight",
                hidden_dim=64,
                head_dim=32,
                temperature=0.35,
                weight_decay=0.0,
            ),
        ]

    return [
        ModelConfig(50, 4, 1, 0.001, 0.2, 32, 45, 8, 0.05, "smote"),
        ModelConfig(50, 4, 2, 0.001, 0.2, 32, 45, 8, 0.05, "smote"),
        ModelConfig(50, 6, 2, 0.001, 0.2, 32, 45, 8, 0.05, "class_weight"),
        ModelConfig(50, 6, 2, 0.001, 0.1, 32, 60, 10, 0.0, "smote", hidden_dim=64, head_dim=32, temperature=0.5, weight_decay=0.0),
        ModelConfig(50, 4, 2, 0.001, 0.1, 32, 60, 10, 0.0, "smote", hidden_dim=64, head_dim=32, temperature=0.5, weight_decay=0.0),
        ModelConfig(50, 6, 2, 0.001, 0.1, 32, 60, 10, 0.02, "class_weight", hidden_dim=64, head_dim=32, temperature=0.35, weight_decay=0.0),
        ModelConfig(50, 6, 2, 0.0005, 0.1, 32, 80, 12, 0.0, "class_weight", hidden_dim=64, head_dim=32, temperature=0.35, weight_decay=0.0),
        ModelConfig(50, 6, 3, 0.001, 0.1, 32, 70, 12, 0.02, "smote", hidden_dim=64, head_dim=32, temperature=0.35, weight_decay=0.0),
    ]


def run_inner_cv_search(
    age: np.ndarray,
    gender: np.ndarray,
    genes: np.ndarray,
    labels: np.ndarray,
    gene_names: list[str],
    quick: bool,
) -> tuple[ModelConfig, pd.DataFrame]:
    """
    Run a stratified inner 5-fold cross-validation hyperparameter search.

    For each candidate configuration in ``build_search_space()``, trains one
    model per fold with ``fit_hybrid_with_validation`` and aggregates the
    fold metrics. The candidates are then ranked and the best configuration
    is returned for final training.

    All preprocessing (feature selection, scaling, class balancing) is applied
    strictly inside each training fold to prevent leakage.

    Returns
    -------
    tuple
        ``(best_config, ranked_results_dataframe)``.
    """
    splitter = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=SEED)
    candidate_results: list[dict[str, float | str | int]] = []
    search_space = build_search_space(quick=quick)

    for candidate_index, config in enumerate(search_space, start=1):
        print(
            f"[{candidate_index}/{len(search_space)}] "
            f"genes={config.n_top_genes}, qubits={config.n_qubits}, "
            f"layers={config.n_layers}, imbalance={config.imbalance_strategy}"
        )
        fold_outcomes: list[TrainingOutcome] = []
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(genes, labels), start=1):
            print(f"  Fold {fold_index}/{INNER_CV_FOLDS}")
            age_train, age_val = age[train_idx], age[val_idx]
            gender_train, gender_val = gender[train_idx], gender[val_idx]
            genes_train, genes_val = genes[train_idx], genes[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            selected_idx, _ = select_genes(
                train_genes=genes_train,
                train_y=y_train,
                gene_names=gene_names,
                n_top_genes=config.n_top_genes,
            )
            X_train = build_selected_matrix(age_train, gender_train, genes_train, selected_idx)
            X_val = build_selected_matrix(age_val, gender_val, genes_val, selected_idx)

            # Refit the scaler inside each fold.
            scaler = fit_scaler(X_train)
            X_train_scaled = scaler.transform(X_train).astype(np.float32)
            X_val_scaled = scaler.transform(X_val).astype(np.float32)

            outcome = fit_hybrid_with_validation(
                X_train=X_train_scaled,
                y_train=y_train,
                X_val=X_val_scaled,
                y_val=y_val,
                config=config,
                seed=SEED + fold_index,
            )
            fold_outcomes.append(outcome)
            print(
                "    "
                f"val_auc={outcome.val_metrics['auc']:.4f}, "
                f"balanced_acc={outcome.val_metrics['balanced_accuracy']:.4f}, "
                f"epoch={outcome.best_epoch}, threshold={outcome.best_threshold:.2f}"
            )

        candidate_results.append(aggregate_fold_results(config=config, fold_results=fold_outcomes))

    ranked = rank_candidate_results(candidate_results)
    best_row = ranked.iloc[0]
    best_config = ModelConfig(
        n_top_genes=int(best_row["n_top_genes"]),
        n_qubits=int(best_row["n_qubits"]),
        n_layers=int(best_row["n_layers"]),
        learning_rate=float(best_row["learning_rate"]),
        dropout=float(best_row["dropout"]),
        batch_size=int(best_row["batch_size"]),
        max_epochs=int(best_row["max_epochs"]),
        patience=int(best_row["patience"]),
        entropy_lambda=float(best_row["entropy_lambda"]),
        imbalance_strategy=str(best_row["imbalance_strategy"]),
        hidden_dim=int(best_row["hidden_dim"]),
        head_dim=int(best_row["head_dim"]),
        temperature=float(best_row["temperature"]),
        weight_decay=float(best_row["weight_decay"]),
    )
    return best_config, ranked


def export_test_patients(
    model: HybridQuantumClassifier,
    X_test_scaled: np.ndarray,
    X_test_unscaled: np.ndarray,
    y_test: np.ndarray,
    selected_gene_names: list[str],
    threshold: float,
) -> list[dict[str, object]]:
    """
    Generate a JSON-serialisable list of holdout patient records for the
    frontend demo and manual inspection.

    Each record includes the patient's demographics, gene expression values
    (unscaled, in log2 space), actual vital status, model-predicted status,
    and predicted mortality probability.
    """
    # Export holdout examples for inspection and demo use.
    probs = infer_probabilities(model, X_test_scaled.astype(np.float32))
    patients: list[dict[str, object]] = []
    for index, row in enumerate(X_test_unscaled):
        genes = {gene: float(value) for gene, value in zip(selected_gene_names, row[2:])}
        patients.append(
            {
                "patient_index": index,
                "age": float(row[0]),
                "gender": "male" if int(row[1]) == 1 else "female",
                "genes": genes,
                "actual_status": "Dead" if int(y_test[index]) == 1 else "Alive",
                "predicted_status": "Dead" if probs[index] >= threshold else "Alive",
                "mortality_probability": float(probs[index]),
            }
        )
    return patients


def summarize_dataset(dataset: Dataset, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, object]:
    """Build a JSON-serialisable summary of the dataset composition and train/holdout split sizes."""
    train_y = dataset.labels[train_idx]
    test_y = dataset.labels[test_idx]
    return {
        "source": "TCGA-GBM (NCI Genomic Data Commons)",
        "raw_expression_shape": {
            "samples": int(dataset.genes.shape[0]),
            "genes": int(dataset.genes.shape[1]),
        },
        "final_labeled_cohort": {
            "samples": int(len(dataset.labels)),
            "dead": int(dataset.labels.sum()),
            "alive": int((dataset.labels == 0).sum()),
            "dead_rate": float(dataset.labels.mean()),
        },
        "split_summary": {
            "train_samples": int(len(train_idx)),
            "holdout_samples": int(len(test_idx)),
            "train_dead": int(train_y.sum()),
            "train_alive": int((train_y == 0).sum()),
            "holdout_dead": int(test_y.sum()),
            "holdout_alive": int((test_y == 0).sum()),
        },
    }


def write_markdown_report(metadata: dict[str, object], cv_results: pd.DataFrame, report_path: Path) -> None:
    """
    Render a human-readable Markdown research report from the training metadata
    and CV results table, and write it to ``report_path``.

    The report includes the task definition, dataset summary, methodology
    description, selected hyperparameters, cross-validation rankings, final
    performance metrics, overfitting analysis, and global explainability.
    """
    dataset_summary = metadata["dataset_summary"]
    selected_hyperparameters = metadata["selected_hyperparameters"]
    holdout_metrics = metadata["holdout_metrics"]
    train_metrics = metadata["train_metrics"]
    auc_ci95 = metadata["holdout_auc_ci95"]

    # Build the markdown training report.
    lines = [
        "# Research-Grade Hybrid Quantum-Classical Training Report",
        "",
        "## Task Definition",
        f"- Task: {metadata['task']}",
        f"- Target definition: {metadata['target_definition']}",
        "- Research framing: Binary mortality-status classification, not time-to-event survival analysis.",
        "",
        "## Dataset Summary",
        f"- Source: {dataset_summary['source']}",
        f"- Final labeled cohort: {dataset_summary['final_labeled_cohort']['samples']} samples",
        f"- Class balance: {dataset_summary['final_labeled_cohort']['dead']} dead / {dataset_summary['final_labeled_cohort']['alive']} alive",
        f"- Holdout split: {dataset_summary['split_summary']['train_samples']} train / {dataset_summary['split_summary']['holdout_samples']} test",
        "",
        "## Methodology Upgrades",
        "- Stratified outer holdout split",
        "- Inner 5-fold cross-validation for model selection",
        "- Fold-local feature selection and scaling",
        "- Leak-free class balancing inside training folds only",
        "- Threshold selection on validation data rather than a fixed 0.5 cutoff",
        "- Overfitting analysis using train-vs-holdout metrics",
        "",
        "## Selected Hyperparameters",
        f"- Top genes: {selected_hyperparameters['n_top_genes']}",
        f"- Qubits: {selected_hyperparameters['n_qubits']}",
        f"- Layers: {selected_hyperparameters['n_layers']}",
        f"- Learning rate: {selected_hyperparameters['learning_rate']}",
        f"- Dropout: {selected_hyperparameters['dropout']}",
        f"- Imbalance strategy: {selected_hyperparameters['imbalance_strategy']}",
        f"- Entropy lambda: {selected_hyperparameters['entropy_lambda']}",
        f"- Final decision threshold: {metadata['decision_threshold']:.3f}",
        f"- Final epochs: {metadata['final_training_epochs']}",
        "",
        "## Cross-Validation Results",
        f"- Best inner-CV mean AUC: {metadata['train_cv_auc']:.4f}",
        f"- Best inner-CV balanced accuracy: {metadata['inner_cv_balanced_accuracy']:.4f}",
        "",
        "Top candidates:",
    ]

    for _, row in cv_results.head(5).iterrows():
        lines.append(
            "- "
            f"genes={int(row['n_top_genes'])}, qubits={int(row['n_qubits'])}, "
            f"layers={int(row['n_layers'])}, imbalance={row['imbalance_strategy']}, "
            f"mean_val_auc={row['mean_val_auc']:.4f}, "
            f"mean_overfit_gap={row['mean_overfit_gap_auc']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Final Performance",
            f"- Train AUC: {train_metrics['auc']:.4f}",
            f"- Holdout AUC: {holdout_metrics['auc']:.4f}",
            f"- Holdout AUC 95% bootstrap CI: [{auc_ci95['lower']:.4f}, {auc_ci95['upper']:.4f}]",
            f"- Holdout PR AUC: {holdout_metrics['pr_auc']:.4f}",
            f"- Holdout accuracy: {holdout_metrics['accuracy']:.4f}",
            f"- Holdout balanced accuracy: {holdout_metrics['balanced_accuracy']:.4f}",
            f"- Holdout F1: {holdout_metrics['f1']:.4f}",
            f"- Holdout precision: {holdout_metrics['precision']:.4f}",
            f"- Holdout recall: {holdout_metrics['recall']:.4f}",
            f"- Holdout specificity: {holdout_metrics['specificity']:.4f}",
            f"- Holdout MCC: {holdout_metrics['mcc']:.4f}",
            f"- Holdout Brier score: {holdout_metrics['brier']:.4f}",
            f"- Risk band cutoffs: low <= {metadata['risk_band_cutoffs']['low_upper']:.3f}, high >= {metadata['risk_band_cutoffs']['high_lower']:.3f}",
            "",
            "## Overfitting Check",
            f"- Train AUC minus holdout AUC: {metadata['overfitting_analysis']['auc_gap']:.4f}",
            f"- Train balanced accuracy minus holdout balanced accuracy: {metadata['overfitting_analysis']['balanced_accuracy_gap']:.4f}",
            "",
            "## Files",
            "- `metadata.json` contains the full machine-readable result summary",
            "- `cv_results.csv` contains the hyperparameter-search table",
            "- `holdout_predictions.csv` contains per-patient holdout predictions",
            "- `test_patients.json` contains randomized UI-ready holdout examples",
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
                "- Global feature importance is the mean absolute Integrated Gradients attribution on the outer holdout cohort.",
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
    report_path.write_text("\n".join(lines))


def main() -> None:
    """
    Entry point for the full research training pipeline.

    Orchestrates: dataset loading → outer holdout split → inner CV search
    → final model training → holdout evaluation → explainability computation
    → artifact export.

    Pass ``--quick`` for a smoke-test run with a reduced search space.
    """
    parser = argparse.ArgumentParser(description="Train the research-grade hybrid quantum-classical model.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller search space for faster smoke testing.",
    )
    args = parser.parse_args()

    set_seeds()
    print("Loading aligned dataset...")
    dataset = load_aligned_dataset()

    indices = np.arange(len(dataset.labels))
    # Create the outer holdout split.
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=dataset.labels,
        random_state=SEED,
    )

    age_train, age_test = dataset.age[train_idx], dataset.age[test_idx]
    gender_train, gender_test = dataset.gender[train_idx], dataset.gender[test_idx]
    genes_train, genes_test = dataset.genes[train_idx], dataset.genes[test_idx]
    y_train, y_test = dataset.labels[train_idx], dataset.labels[test_idx]

    print("Running inner cross-validation hyperparameter search...")
    best_config, cv_results = run_inner_cv_search(
        age=age_train,
        gender=gender_train,
        genes=genes_train,
        labels=y_train,
        gene_names=dataset.gene_names,
        quick=args.quick,
    )
    best_result = cv_results.iloc[0]

    # Refit feature selection on the full training split.
    selected_idx, selected_gene_names = select_genes(
        train_genes=genes_train,
        train_y=y_train,
        gene_names=dataset.gene_names,
        n_top_genes=best_config.n_top_genes,
    )
    X_train_selected = build_selected_matrix(age_train, gender_train, genes_train, selected_idx)
    X_test_selected = build_selected_matrix(age_test, gender_test, genes_test, selected_idx)

    scaler = fit_scaler(X_train_selected)
    X_train_scaled = scaler.transform(X_train_selected).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_selected).astype(np.float32)
    # Use the median training profile as the explainability baseline.
    reference_unscaled = np.median(X_train_selected, axis=0).astype(np.float32)
    reference_scaled = scaler.transform(reference_unscaled.reshape(1, -1)).astype(np.float32)[0]

    final_epochs = int(max(1, round(float(best_result["mean_best_epoch"]))))
    decision_threshold = float(best_result["mean_best_threshold"])

    print("Training final hybrid model on the full training split...")
    final_model = fit_final_model(
        X_train=X_train_scaled,
        y_train=y_train,
        config=best_config,
        fixed_epochs=final_epochs,
        seed=SEED,
    )
    train_probs = infer_probabilities(final_model, X_train_scaled)
    holdout_probs = infer_probabilities(final_model, X_test_scaled)

    train_metrics = evaluate_predictions(y_train, train_probs, decision_threshold)
    holdout_metrics = evaluate_predictions(y_test, holdout_probs, decision_threshold)
    auc_ci95 = bootstrap_auc_ci(y_test, holdout_probs)
    risk_band_cutoffs = compute_risk_band_cutoffs(train_probs, decision_threshold)
    feature_order = ["age", "gender"] + selected_gene_names
    # Compute explainability on holdout samples.
    holdout_attributions = shap_explain(
        model=final_model,
        inputs=X_test_scaled,
        baseline=reference_scaled,
        device="cpu",
    )
    explainability = build_global_explainability(
        feature_names=feature_order,
        attributions=holdout_attributions,
    )

    overfitting_analysis = {
        "auc_gap": float(train_metrics["auc"] - holdout_metrics["auc"]),
        "balanced_accuracy_gap": float(
            train_metrics["balanced_accuracy"] - holdout_metrics["balanced_accuracy"]
        ),
        "interpretation": (
            "potential_overfitting"
            if (train_metrics["auc"] - holdout_metrics["auc"]) > 0.1
            else "acceptable_generalization"
        ),
    }

    model_path = ARTIFACT_DIR / "hybrid_model_state.pt"
    preprocess_path = ARTIFACT_DIR / "preprocessing.joblib"
    metadata_path = ARTIFACT_DIR / "metadata.json"
    samples_path = ARTIFACT_DIR / "test_patients.json"
    cv_results_path = ARTIFACT_DIR / "cv_results.csv"
    holdout_predictions_path = ARTIFACT_DIR / "holdout_predictions.csv"
    report_path = ARTIFACT_DIR / "research_report.md"
    explainability_path = ARTIFACT_DIR / "explainability.json"

    # Save model and preprocessing artifacts.
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "model_params": final_model.get_init_params(),
            "n_features": int(X_train_scaled.shape[1]),
        },
        model_path,
    )
    joblib.dump(
        {
            "scaler": scaler,
            "selected_genes": selected_gene_names,
            "feature_order": feature_order,
            "reference_unscaled": reference_unscaled.tolist(),
            "reference_scaled": reference_scaled.tolist(),
        },
        preprocess_path,
    )

    holdout_predictions = pd.DataFrame(
        {
            "patient_index": np.arange(len(y_test)),
            "actual_status": np.where(y_test == 1, "Dead", "Alive"),
            "predicted_status": np.where(holdout_probs >= decision_threshold, "Dead", "Alive"),
            "mortality_probability": holdout_probs,
            "age": age_test,
            "gender": np.where(gender_test == 1, "male", "female"),
        }
    )
    holdout_predictions.to_csv(holdout_predictions_path, index=False)
    cv_results.to_csv(cv_results_path, index=False)

    metadata = {
        "task": "GBM mortality-status classification",
        "target_definition": "1 = Dead, 0 = Alive",
        "research_positioning": "Hybrid quantum-classical binary classification on TCGA-GBM. Not time-to-event survival analysis.",
        "selected_model": "hybrid_quantum_classical",
        "dataset_summary": summarize_dataset(dataset, train_idx=train_idx, test_idx=test_idx),
        "preprocessing": {
            "expression_transform": "log2(TPM + 1)",
            "clinical_features": ["age", "gender"],
            "feature_selection": "SelectKBest(f_classif) fitted on training data only",
            "scaling": "MinMaxScaler fitted on training data only",
            "class_balancing_policy": "Applied only within training folds or final training split",
        },
        "validation_protocol": {
            "outer_split": "Stratified train/test split (80/20)",
            "inner_cv": f"Stratified {INNER_CV_FOLDS}-fold cross-validation on the training partition",
            "selection_metric": "mean validation AUC",
            "threshold_selection": "Validation-balanced-accuracy threshold",
        },
        "selected_hyperparameters": asdict(best_config),
        "train_cv_auc": float(best_result["mean_val_auc"]),
        "inner_cv_balanced_accuracy": float(best_result["mean_val_balanced_accuracy"]),
        "inner_cv_results_top5": cv_results.head(5).to_dict(orient="records"),
        "decision_threshold": decision_threshold,
        "final_training_epochs": final_epochs,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_auc_ci95": auc_ci95,
        "risk_band_cutoffs": risk_band_cutoffs,
        "overfitting_analysis": overfitting_analysis,
        "selected_genes": selected_gene_names,
        "feature_order": feature_order,
        "explainability": {
            "local_method": "Integrated Gradients relative to the median training-cohort feature profile",
            "global_method": "Mean absolute Integrated Gradients attribution on the outer holdout cohort",
            "baseline_reference": "Median training-cohort feature vector",
            "top_global_features": explainability["feature_importance"][:10],
        },
        "artifact_files": {
            "model": model_path.name,
            "preprocessing": preprocess_path.name,
            "metadata": metadata_path.name,
            "test_patients": samples_path.name,
            "cv_results": cv_results_path.name,
            "holdout_predictions": holdout_predictions_path.name,
            "research_report": report_path.name,
            "explainability": explainability_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    explainability_path.write_text(json.dumps(explainability, indent=2))

    test_patients = export_test_patients(
        model=final_model,
        X_test_scaled=X_test_scaled,
        X_test_unscaled=X_test_selected,
        y_test=y_test,
        selected_gene_names=selected_gene_names,
        threshold=decision_threshold,
    )
    samples_path.write_text(json.dumps(test_patients, indent=2))
    write_markdown_report(metadata=metadata, cv_results=cv_results, report_path=report_path)

    print(f"Saved model state to {model_path}")
    print(f"Saved preprocessing to {preprocess_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved CV results to {cv_results_path}")
    print(f"Saved holdout predictions to {holdout_predictions_path}")
    print(f"Saved research report to {report_path}")
    print(f"Saved explainability report to {explainability_path}")
    print(f"Saved test patients to {samples_path}")
    print("Selected configuration:")
    print(json.dumps(asdict(best_config), indent=2))
    print("Holdout metrics:")
    for key, value in holdout_metrics.items():
        if key == "confusion_matrix":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
