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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qubrain.backend.app.explainability import build_global_explainability, integrated_gradients

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "qubrain" / "backend" / "model_artifacts_survival"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
INNER_CV_FOLDS = 5
FINAL_BOOTSTRAP_SAMPLES = 500
DEVICE = torch.device("cpu")
FINAL_ENSEMBLE_SEEDS = [42, 1337, 2026, 7, 99]
SURVIVAL_CATEGORICAL_COLUMNS = [
    "site_of_resection_or_biopsy",
    "prior_malignancy",
    "synchronous_malignancy",
]
UNKNOWN_CATEGORY_TOKENS = {
    "",
    "--",
    "not reported",
    "not applicable",
    "unknown",
    "nan",
    "none",
}


@dataclass(frozen=True)
class SurvivalModelConfig:
    n_top_genes: int
    n_qubits: int
    n_layers: int
    learning_rate: float
    dropout: float
    max_epochs: int
    patience: int
    hidden_dim: int = 32
    head_dim: int = 16
    weight_decay: float = 1e-4
    ranking_lambda: float = 0.3
    min_category_count: int = 8


@dataclass
class SurvivalDataset:
    age: np.ndarray
    gender: np.ndarray
    categorical_clinical: pd.DataFrame
    genes: np.ndarray
    gene_names: list[str]
    time_days: np.ndarray
    event: np.ndarray


@dataclass
class SurvivalTrainingOutcome:
    model: "HybridQuantumSurvivalModel"
    train_risk: np.ndarray
    val_risk: np.ndarray
    best_epoch: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]


@dataclass
class CategoryEncoder:
    categories: dict[str, list[str]]
    feature_names: list[str]


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_single_gene_file(filepath: Path) -> pd.Series | None:
    try:
        df = pd.read_csv(filepath, sep="\t", skiprows=1, low_memory=False)
        if "gene_name" not in df.columns or "tpm_unstranded" not in df.columns:
            return None
        if "gene_type" in df.columns:
            df = df[df["gene_type"] == "protein_coding"]
        return df[["gene_name", "tpm_unstranded"]].dropna().set_index("gene_name")["tpm_unstranded"]
    except Exception as exc:  # pragma: no cover
        print(f"Failed to read {filepath}: {exc}")
        return None


def load_expression_matrix() -> pd.DataFrame:
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

    matrix = pd.DataFrame(samples).dropna(how="all")
    return np.log2(matrix.T + 1)


def _first_valid(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _max_valid(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    return values.max()


def _mode_or_first(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    modes = values.mode()
    if not modes.empty:
        return modes.iloc[0]
    return values.iloc[0]


def normalize_category_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    normalized = str(value).strip().lower()
    if normalized in UNKNOWN_CATEGORY_TOKENS:
        return "unknown"
    return normalized.replace("/", "_").replace(" ", "_")


def fit_category_encoder(category_frame: pd.DataFrame, min_count: int) -> CategoryEncoder:
    categories: dict[str, list[str]] = {}
    feature_names: list[str] = []

    for column in category_frame.columns:
        normalized = category_frame[column].map(normalize_category_value)
        value_counts = normalized.value_counts()
        frequent = [value for value, count in value_counts.items() if count >= min_count and value != "unknown"]
        frequent = sorted(frequent)

        column_categories = frequent + ["other", "unknown"]
        categories[column] = column_categories
        feature_names.extend([f"{column}={value}" for value in column_categories])

    return CategoryEncoder(categories=categories, feature_names=feature_names)


def transform_category_frame(category_frame: pd.DataFrame, encoder: CategoryEncoder) -> np.ndarray:
    blocks: list[np.ndarray] = []

    for column in category_frame.columns:
        column_values = category_frame[column].map(normalize_category_value)
        known_values = set(encoder.categories[column])
        frequent_values = known_values - {"other", "unknown"}
        transformed = []

        for value in column_values:
            if value == "unknown":
                transformed.append("unknown")
            elif value in frequent_values:
                transformed.append(value)
            else:
                transformed.append("other")

        matrix = np.column_stack(
            [(np.asarray(transformed) == category).astype(np.float32) for category in encoder.categories[column]]
        )
        blocks.append(matrix)

    if not blocks:
        return np.zeros((len(category_frame), 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32)


def load_survival_clinical_data() -> pd.DataFrame:
    clinical_file = DATA_DIR / "clinical.project-tcga-gbm.2026-01-08" / "clinical.tsv"
    df = pd.read_csv(clinical_file, sep="\t", low_memory=False)

    columns = {
        "cases.case_id": "case_id",
        "demographic.age_at_index": "age",
        "demographic.gender": "gender",
        "demographic.vital_status": "vital_status",
        "demographic.days_to_death": "days_to_death",
        "diagnoses.days_to_last_follow_up": "days_to_last_follow_up",
        "diagnoses.site_of_resection_or_biopsy": "site_of_resection_or_biopsy",
        "diagnoses.prior_malignancy": "prior_malignancy",
        "diagnoses.synchronous_malignancy": "synchronous_malignancy",
    }
    df = df[list(columns.keys())].rename(columns=columns)

    for numeric_col in ["age", "days_to_death", "days_to_last_follow_up"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    grouped = (
        df.groupby("case_id", as_index=False)
        .agg(
            {
                "age": _max_valid,
                "gender": _mode_or_first,
                "vital_status": _mode_or_first,
                "days_to_death": _max_valid,
                "days_to_last_follow_up": _max_valid,
                "site_of_resection_or_biopsy": _mode_or_first,
                "prior_malignancy": _mode_or_first,
                "synchronous_malignancy": _mode_or_first,
            }
        )
        .copy()
    )

    grouped = grouped[grouped["vital_status"].isin(["Alive", "Dead"])].copy()
    grouped["event"] = (grouped["vital_status"] == "Dead").astype(int)
    grouped["time_days"] = np.where(
        grouped["event"] == 1,
        grouped["days_to_death"].fillna(grouped["days_to_last_follow_up"]),
        grouped["days_to_last_follow_up"],
    )

    grouped["gender"] = grouped["gender"].str.lower().map({"male": 1, "female": 0})
    grouped["age"] = grouped["age"].fillna(grouped["age"].median())
    grouped["gender"] = grouped["gender"].fillna(grouped["gender"].mode().iloc[0]).astype(int)
    grouped = grouped[grouped["time_days"].notna()].copy()
    grouped = grouped[grouped["time_days"] > 0].reset_index(drop=True)
    return grouped[
        [
            "case_id",
            "age",
            "gender",
            "event",
            "time_days",
            "site_of_resection_or_biopsy",
            "prior_malignancy",
            "synchronous_malignancy",
        ]
    ]


def load_aligned_survival_dataset() -> SurvivalDataset:
    expression = load_expression_matrix()
    clinical = load_survival_clinical_data()

    mapping_file = DATA_DIR / "file_case_mapping.csv"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_file}")

    mapping = pd.read_csv(mapping_file)
    aligned = clinical.merge(mapping, on="case_id", how="inner")
    aligned = aligned[aligned["file_id"].isin(expression.index)].copy()
    aligned = aligned.drop_duplicates(subset=["file_id"]).reset_index(drop=True)

    expression_aligned = expression.loc[aligned["file_id"]]
    return SurvivalDataset(
        age=aligned["age"].to_numpy(dtype=float),
        gender=aligned["gender"].to_numpy(dtype=int),
        categorical_clinical=aligned[SURVIVAL_CATEGORICAL_COLUMNS].reset_index(drop=True),
        genes=expression_aligned.to_numpy(dtype=float),
        gene_names=expression_aligned.columns.tolist(),
        time_days=aligned["time_days"].to_numpy(dtype=float),
        event=aligned["event"].to_numpy(dtype=int),
    )


def select_genes_for_survival(
    train_genes: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    gene_names: list[str],
    n_top_genes: int,
) -> tuple[np.ndarray, list[str]]:
    non_constant_indices = np.where(np.var(train_genes, axis=0) > 0)[0]
    filtered = train_genes[:, non_constant_indices].astype(np.float64)

    order = np.argsort(-train_time, kind="mergesort")
    x_sorted = filtered[order]
    event_sorted = train_event[order].astype(bool)

    risk_sum = np.cumsum(x_sorted, axis=0)
    risk_count = np.arange(1, len(train_time) + 1, dtype=np.float64).reshape(-1, 1)
    risk_mean = risk_sum / risk_count

    event_rows = np.where(event_sorted)[0]
    if len(event_rows) == 0:
        raise ValueError("Survival feature selection requires at least one observed event.")

    score = np.abs(np.sum(x_sorted[event_rows] - risk_mean[event_rows], axis=0))
    score = score / (np.std(filtered, axis=0) + 1e-8)

    k = min(n_top_genes, score.shape[0])
    selected_local = np.argsort(score)[-k:][::-1]
    selected_idx = non_constant_indices[selected_local]
    selected_names = [gene_names[index] for index in selected_idx]
    return selected_idx, selected_names


def build_feature_matrix(
    age: np.ndarray,
    gender: np.ndarray,
    categorical_matrix: np.ndarray,
    genes: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    clinical = np.column_stack([age, gender])
    selected_genes = genes[:, selected_idx]
    pieces = [clinical]
    if categorical_matrix.size:
        pieces.append(categorical_matrix)
    pieces.append(selected_genes)
    return np.hstack(pieces).astype(np.float32)


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def make_survival_strata(event: np.ndarray, time_days: np.ndarray, n_bins: int = 4) -> np.ndarray:
    time_series = pd.Series(time_days)
    quantile_bins = pd.qcut(time_series.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    if quantile_bins.isna().any():
        quantile_bins = quantile_bins.fillna(0)
    return np.array([f"{int(evt)}_{int(bin_id)}" for evt, bin_id in zip(event, quantile_bins)], dtype=object)


def concordance_index(time_days: np.ndarray, event: np.ndarray, risk_scores: np.ndarray) -> float:
    comparable = 0
    concordant = 0.0
    n_samples = len(time_days)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if time_days[i] == time_days[j]:
                continue

            if event[i] == 1 and time_days[i] < time_days[j]:
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
            elif event[j] == 1 and time_days[j] < time_days[i]:
                comparable += 1
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1.0
                elif risk_scores[j] == risk_scores[i]:
                    concordant += 0.5

    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def bootstrap_c_index_ci(
    time_days: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    n_bootstrap: int = FINAL_BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(time_days))
    values: list[float] = []

    for _ in range(n_bootstrap):
        chosen = rng.choice(indices, size=len(indices), replace=True)
        c_index = concordance_index(time_days[chosen], event[chosen], risk_scores[chosen])
        if not np.isnan(c_index):
            values.append(c_index)

    if not values:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}

    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(array, 0.025)),
        "upper": float(np.quantile(array, 0.975)),
    }


def compute_risk_band_cutoffs(risk_scores: np.ndarray) -> dict[str, float]:
    low_upper = float(np.quantile(risk_scores, 0.33))
    high_lower = float(np.quantile(risk_scores, 0.67))

    if high_lower <= low_upper:
        midpoint = float(np.median(risk_scores))
        low_upper = midpoint - 0.1
        high_lower = midpoint + 0.1

    return {"low_upper": low_upper, "high_lower": high_lower}


def assign_risk_band(risk_scores: np.ndarray, cutoffs: dict[str, float]) -> np.ndarray:
    bands = np.full(len(risk_scores), "moderate", dtype=object)
    bands[risk_scores <= cutoffs["low_upper"]] = "low"
    bands[risk_scores >= cutoffs["high_lower"]] = "high"
    return bands


class CoxPartialLikelihoodLoss(nn.Module):
    def __init__(self, ranking_lambda: float = 0.0) -> None:
        super().__init__()
        self.ranking_lambda = ranking_lambda

    def _cox_loss(
        self,
        risk_scores: torch.Tensor,
        time_days: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        order = torch.argsort(time_days, descending=True)
        ordered_risk = risk_scores[order]
        ordered_event = event[order]

        log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
        event_mask = ordered_event > 0.5
        if torch.sum(event_mask) == 0:
            raise ValueError("Cox training requires at least one observed event in the training split.")

        partial_log_likelihood = ordered_risk[event_mask] - log_risk_set[event_mask]
        return -torch.mean(partial_log_likelihood)

    def _pairwise_ranking_loss(
        self,
        risk_scores: torch.Tensor,
        time_days: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        time_i = time_days.unsqueeze(1)
        time_j = time_days.unsqueeze(0)
        event_i = event.unsqueeze(1)
        comparable = (event_i > 0.5) & (time_i < time_j)

        if not torch.any(comparable):
            return torch.zeros((), device=risk_scores.device, dtype=risk_scores.dtype)

        risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
        pairwise_losses = torch.nn.functional.softplus(-risk_diff)
        return torch.mean(pairwise_losses[comparable])

    def forward(
        self,
        risk_scores: torch.Tensor,
        time_days: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        cox_loss = self._cox_loss(risk_scores, time_days, event)
        if self.ranking_lambda <= 0:
            return cox_loss

        ranking_loss = self._pairwise_ranking_loss(risk_scores, time_days, event)
        return cox_loss + (self.ranking_lambda * ranking_loss)


class HybridQuantumSurvivalModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_qubits: int,
        n_layers: int,
        hidden_dim: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout = dropout

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
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        quantum_input = (encoded + 1) * (math.pi / 2)
        quantum_output = self.q_layer(quantum_input)
        risk_score = self.head(quantum_output)
        return risk_score.squeeze(-1)


def instantiate_model(
    config: SurvivalModelConfig,
    n_features: int,
) -> HybridQuantumSurvivalModel:
    return HybridQuantumSurvivalModel(
        n_features=n_features,
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        head_dim=config.head_dim,
        dropout=config.dropout,
    )


def infer_risk_scores(model: HybridQuantumSurvivalModel, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        return model(tensor).cpu().numpy()


def evaluate_survival_predictions(
    time_days: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
) -> dict[str, float]:
    return {
        "c_index": concordance_index(time_days, event, risk_scores),
        "mean_risk_score": float(np.mean(risk_scores)),
        "risk_score_std": float(np.std(risk_scores)),
    }


def fit_hybrid_survival_with_validation(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_val: np.ndarray,
    time_val: np.ndarray,
    event_val: np.ndarray,
    config: SurvivalModelConfig,
) -> SurvivalTrainingOutcome:
    model = instantiate_model(
        config=config,
        n_features=X_train.shape[1],
    ).to(DEVICE)
    criterion = CoxPartialLikelihoodLoss(ranking_lambda=config.ranking_lambda)
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
    time_train_t = torch.from_numpy(time_train.astype(np.float32)).to(DEVICE)
    event_train_t = torch.from_numpy(event_train.astype(np.float32)).to(DEVICE)

    best_c_index = -1.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        risk_scores = model(X_train_t)
        loss = criterion(risk_scores, time_train_t, event_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_risk = infer_risk_scores(model, X_val)
        val_c_index = concordance_index(time_val, event_val, val_risk)
        scheduler.step(val_c_index)

        if val_c_index > best_c_index:
            best_c_index = val_c_index
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is None:  # pragma: no cover
        raise RuntimeError("Training failed to produce a valid survival model state.")

    model.load_state_dict(best_state)
    train_risk = infer_risk_scores(model, X_train)
    val_risk = infer_risk_scores(model, X_val)

    return SurvivalTrainingOutcome(
        model=model,
        train_risk=train_risk,
        val_risk=val_risk,
        best_epoch=best_epoch,
        train_metrics=evaluate_survival_predictions(time_train, event_train, train_risk),
        val_metrics=evaluate_survival_predictions(time_val, event_val, val_risk),
    )


def fit_final_survival_model(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    config: SurvivalModelConfig,
    fixed_epochs: int,
) -> HybridQuantumSurvivalModel:
    model = instantiate_model(
        config=config,
        n_features=X_train.shape[1],
    ).to(DEVICE)
    criterion = CoxPartialLikelihoodLoss(ranking_lambda=config.ranking_lambda)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    time_train_t = torch.from_numpy(time_train.astype(np.float32)).to(DEVICE)
    event_train_t = torch.from_numpy(event_train.astype(np.float32)).to(DEVICE)

    for _ in range(max(1, fixed_epochs)):
        model.train()
        optimizer.zero_grad()
        risk_scores = model(X_train_t)
        loss = criterion(risk_scores, time_train_t, event_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    return model


def fit_final_survival_ensemble(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    config: SurvivalModelConfig,
    fixed_epochs: int,
    seeds: list[int],
) -> list[HybridQuantumSurvivalModel]:
    models: list[HybridQuantumSurvivalModel] = []
    for seed in seeds:
        set_seeds(seed)
        model = fit_final_survival_model(
            X_train=X_train,
            time_train=time_train,
            event_train=event_train,
            config=config,
            fixed_epochs=fixed_epochs,
        )
        models.append(model)
    return models


def infer_ensemble_risk_scores(models: list[HybridQuantumSurvivalModel], X: np.ndarray) -> np.ndarray:
    risks = [infer_risk_scores(model, X) for model in models]
    return np.mean(np.vstack(risks), axis=0)


def integrated_gradients_ensemble(
    models: list[HybridQuantumSurvivalModel],
    inputs: np.ndarray,
    baseline: np.ndarray,
    steps: int = 24,
    device: str = "cpu",
) -> np.ndarray:
    all_attributions = [
        integrated_gradients(
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
    config: SurvivalModelConfig,
    fold_results: list[SurvivalTrainingOutcome],
) -> dict[str, float | int]:
    mean_train_c_index = float(np.mean([result.train_metrics["c_index"] for result in fold_results]))
    mean_val_c_index = float(np.mean([result.val_metrics["c_index"] for result in fold_results]))
    mean_best_epoch = float(np.mean([result.best_epoch for result in fold_results]))

    return {
        **asdict(config),
        "mean_train_c_index": mean_train_c_index,
        "mean_val_c_index": mean_val_c_index,
        "mean_overfit_gap_c_index": mean_train_c_index - mean_val_c_index,
        "mean_best_epoch": mean_best_epoch,
    }


def rank_candidate_results(results: list[dict[str, float | int]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df.sort_values(
        by=["mean_val_c_index", "mean_overfit_gap_c_index"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_search_space(quick: bool) -> list[SurvivalModelConfig]:
    if quick:
        return [
            SurvivalModelConfig(
                n_top_genes=50,
                n_qubits=4,
                n_layers=2,
                learning_rate=0.001,
                dropout=0.2,
                max_epochs=16,
                patience=5,
                ranking_lambda=0.2,
            ),
            SurvivalModelConfig(
                n_top_genes=50,
                n_qubits=6,
                n_layers=2,
                learning_rate=0.001,
                dropout=0.1,
                max_epochs=20,
                patience=6,
                hidden_dim=64,
                head_dim=32,
                weight_decay=0.0,
                ranking_lambda=0.3,
            ),
        ]

    return [
        SurvivalModelConfig(50, 4, 2, 0.0010, 0.25, 32, 8, ranking_lambda=0.1, min_category_count=8),
        SurvivalModelConfig(50, 4, 2, 0.0005, 0.20, 40, 10, hidden_dim=64, head_dim=32, weight_decay=0.0, ranking_lambda=0.3, min_category_count=6),
        SurvivalModelConfig(50, 6, 2, 0.0010, 0.15, 40, 10, hidden_dim=64, head_dim=32, weight_decay=0.0, ranking_lambda=0.3, min_category_count=6),
        SurvivalModelConfig(100, 4, 2, 0.0005, 0.20, 40, 10, hidden_dim=64, head_dim=32, weight_decay=0.0, ranking_lambda=0.3, min_category_count=6),
        SurvivalModelConfig(100, 6, 2, 0.0005, 0.15, 48, 12, hidden_dim=64, head_dim=32, weight_decay=0.0, ranking_lambda=0.4, min_category_count=6),
        SurvivalModelConfig(150, 4, 2, 0.0005, 0.20, 48, 12, hidden_dim=64, head_dim=32, weight_decay=0.0, ranking_lambda=0.4, min_category_count=6),
        SurvivalModelConfig(100, 6, 3, 0.0005, 0.15, 52, 12, hidden_dim=96, head_dim=48, weight_decay=0.0, ranking_lambda=0.4, min_category_count=6),
        SurvivalModelConfig(150, 6, 3, 0.0003, 0.10, 60, 14, hidden_dim=96, head_dim=48, weight_decay=0.0, ranking_lambda=0.5, min_category_count=5),
    ]


def run_inner_cv_search(
    age: np.ndarray,
    gender: np.ndarray,
    categorical_clinical: pd.DataFrame,
    genes: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    gene_names: list[str],
    quick: bool,
) -> tuple[SurvivalModelConfig, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=SEED)
    strata = make_survival_strata(event, time_days)
    candidate_results: list[dict[str, float | int]] = []

    for config in build_search_space(quick=quick):
        print(
            "Evaluating survival config "
            f"genes={config.n_top_genes}, qubits={config.n_qubits}, layers={config.n_layers}"
        )
        fold_outcomes: list[SurvivalTrainingOutcome] = []

        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(genes, strata), start=1):
            print(f"  Fold {fold_index}/{INNER_CV_FOLDS}")
            selected_idx, _ = select_genes_for_survival(
                train_genes=genes[train_idx],
                train_time=time_days[train_idx],
                train_event=event[train_idx],
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

            outcome = fit_hybrid_survival_with_validation(
                X_train=X_train_scaled,
                time_train=time_days[train_idx],
                event_train=event[train_idx],
                X_val=X_val_scaled,
                time_val=time_days[val_idx],
                event_val=event[val_idx],
                config=config,
            )
            fold_outcomes.append(outcome)
            print(
                "    "
                f"val_c_index={outcome.val_metrics['c_index']:.4f}, "
                f"epoch={outcome.best_epoch}"
            )

        candidate_results.append(aggregate_fold_results(config=config, fold_results=fold_outcomes))

    ranked = rank_candidate_results(candidate_results)
    best_row = ranked.iloc[0]
    best_config = SurvivalModelConfig(
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
        ranking_lambda=float(best_row["ranking_lambda"]),
        min_category_count=int(best_row["min_category_count"]),
    )
    return best_config, ranked


def summarize_dataset(dataset: SurvivalDataset, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, object]:
    train_event = dataset.event[train_idx]
    test_event = dataset.event[test_idx]
    return {
        "source": "TCGA-GBM (NCI Genomic Data Commons)",
        "raw_expression_shape": {
            "samples": int(dataset.genes.shape[0]),
            "genes": int(dataset.genes.shape[1]),
        },
        "final_survival_cohort": {
            "samples": int(len(dataset.event)),
            "observed_events": int(dataset.event.sum()),
            "censored": int((dataset.event == 0).sum()),
            "event_rate": float(dataset.event.mean()),
            "median_time_days": float(np.median(dataset.time_days)),
        },
        "split_summary": {
            "train_samples": int(len(train_idx)),
            "holdout_samples": int(len(test_idx)),
            "train_events": int(train_event.sum()),
            "train_censored": int((train_event == 0).sum()),
            "holdout_events": int(test_event.sum()),
            "holdout_censored": int((test_event == 0).sum()),
        },
    }


def write_markdown_report(metadata: dict[str, object], cv_results: pd.DataFrame, report_path: Path) -> None:
    dataset_summary = metadata["dataset_summary"]
    selected_hyperparameters = metadata["selected_hyperparameters"]
    holdout_metrics = metadata["holdout_metrics"]
    train_metrics = metadata["train_metrics"]
    c_index_ci95 = metadata["holdout_c_index_ci95"]

    lines = [
        "# Hybrid Quantum-Classical Survival Training Report",
        "",
        "## Task Definition",
        f"- Task: {metadata['task']}",
        f"- Target definition: {metadata['target_definition']}",
        "- Research framing: Time-to-event survival modeling with censoring.",
        "",
        "## Dataset Summary",
        f"- Source: {dataset_summary['source']}",
        f"- Final survival cohort: {dataset_summary['final_survival_cohort']['samples']} samples",
        f"- Observed events: {dataset_summary['final_survival_cohort']['observed_events']}",
        f"- Censored cases: {dataset_summary['final_survival_cohort']['censored']}",
        f"- Median observed time: {dataset_summary['final_survival_cohort']['median_time_days']:.1f} days",
        "",
        "## Methodology",
        "- Stratified outer holdout split on event status",
        "- Inner 5-fold cross-validation for model selection",
        "- Fold-local survival-aware feature ranking",
        "- Fold-local one-hot encoding of baseline categorical clinical covariates",
        "- Fold-local scaling",
        "- Hybrid quantum-classical Cox-plus-ranking survival training objective",
        "",
        "## Selected Hyperparameters",
        f"- Top genes: {selected_hyperparameters['n_top_genes']}",
        f"- Qubits: {selected_hyperparameters['n_qubits']}",
        f"- Layers: {selected_hyperparameters['n_layers']}",
        f"- Learning rate: {selected_hyperparameters['learning_rate']}",
        f"- Dropout: {selected_hyperparameters['dropout']}",
        f"- Ranking lambda: {selected_hyperparameters['ranking_lambda']}",
        f"- Final epochs: {metadata['final_training_epochs']}",
        f"- Ensemble size: {len(metadata['ensemble_seeds'])}",
        "",
        "## Cross-Validation Results",
        f"- Best inner-CV mean C-index: {metadata['train_cv_c_index']:.4f}",
        "Top candidates:",
    ]

    for _, row in cv_results.head(5).iterrows():
        lines.append(
            "- "
            f"genes={int(row['n_top_genes'])}, qubits={int(row['n_qubits'])}, "
            f"layers={int(row['n_layers'])}, mean_val_c_index={row['mean_val_c_index']:.4f}, "
            f"mean_overfit_gap={row['mean_overfit_gap_c_index']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Final Performance",
            f"- Train C-index: {train_metrics['c_index']:.4f}",
            f"- Holdout C-index: {holdout_metrics['c_index']:.4f}",
            f"- Holdout C-index 95% bootstrap CI: [{c_index_ci95['lower']:.4f}, {c_index_ci95['upper']:.4f}]",
            f"- Risk band cutoffs: low <= {metadata['risk_band_cutoffs']['low_upper']:.4f}, high >= {metadata['risk_band_cutoffs']['high_lower']:.4f}",
            "",
            "## Interpretation Guardrail",
            "- This model outputs relative survival risk scores rather than exact death dates.",
            "- Patient-facing month ranges should be derived later from calibrated survival curves or risk-group Kaplan-Meier summaries.",
            "",
            "## Files",
            "- `metadata.json` contains the machine-readable result summary",
            "- `cv_results.csv` contains the survival hyperparameter search table",
            "- `holdout_predictions.csv` contains per-patient holdout risk predictions",
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

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid quantum-classical survival model.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller hyperparameter search for a faster smoke-test pass.",
    )
    args = parser.parse_args()

    set_seeds()
    print("Loading aligned survival dataset...")
    dataset = load_aligned_survival_dataset()

    indices = np.arange(len(dataset.event))
    split_strata = make_survival_strata(dataset.event, dataset.time_days)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=split_strata,
        random_state=SEED,
    )

    age_train, age_test = dataset.age[train_idx], dataset.age[test_idx]
    gender_train, gender_test = dataset.gender[train_idx], dataset.gender[test_idx]
    categorical_train = dataset.categorical_clinical.iloc[train_idx].reset_index(drop=True)
    categorical_test = dataset.categorical_clinical.iloc[test_idx].reset_index(drop=True)
    genes_train, genes_test = dataset.genes[train_idx], dataset.genes[test_idx]
    time_train, time_test = dataset.time_days[train_idx], dataset.time_days[test_idx]
    event_train, event_test = dataset.event[train_idx], dataset.event[test_idx]

    print("Running inner cross-validation hyperparameter search...")
    best_config, cv_results = run_inner_cv_search(
        age=age_train,
        gender=gender_train,
        categorical_clinical=categorical_train,
        genes=genes_train,
        time_days=time_train,
        event=event_train,
        gene_names=dataset.gene_names,
        quick=args.quick,
    )
    best_result = cv_results.iloc[0]

    selected_idx, selected_gene_names = select_genes_for_survival(
        train_genes=genes_train,
        train_time=time_train,
        train_event=event_train,
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

    final_epochs = int(max(1, round(float(best_result["mean_best_epoch"]))))
    print("Training final hybrid survival ensemble on the full training split...")
    final_models = fit_final_survival_ensemble(
        X_train=X_train_scaled,
        time_train=time_train,
        event_train=event_train,
        config=best_config,
        fixed_epochs=final_epochs,
        seeds=FINAL_ENSEMBLE_SEEDS,
    )

    train_risk = infer_ensemble_risk_scores(final_models, X_train_scaled)
    holdout_risk = infer_ensemble_risk_scores(final_models, X_test_scaled)
    train_metrics = evaluate_survival_predictions(time_train, event_train, train_risk)
    holdout_metrics = evaluate_survival_predictions(time_test, event_test, holdout_risk)
    c_index_ci95 = bootstrap_c_index_ci(time_test, event_test, holdout_risk)
    risk_band_cutoffs = compute_risk_band_cutoffs(train_risk)

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

    model_path = ARTIFACT_DIR / "hybrid_survival_model_state.pt"
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
            "risk_band_cutoffs": risk_band_cutoffs,
        },
        preprocess_path,
    )

    holdout_bands = assign_risk_band(holdout_risk, risk_band_cutoffs)
    holdout_predictions = pd.DataFrame(
        {
            "patient_index": np.arange(len(event_test)),
            "event_observed": event_test,
            "survival_days": time_test,
            "survival_months": time_test / 30.44,
            "risk_score": holdout_risk,
            "risk_band": holdout_bands,
            "age": age_test,
            "gender": np.where(gender_test == 1, "male", "female"),
        }
    )
    holdout_predictions.to_csv(holdout_predictions_path, index=False)
    cv_results.to_csv(cv_results_path, index=False)

    metadata = {
        "task": "GBM time-to-event survival modeling",
        "target_definition": "time = days_to_death or days_to_last_follow_up, event = 1 if dead else 0",
        "research_positioning": "Hybrid quantum-classical Cox-style survival modeling on TCGA-GBM with censoring.",
        "selected_model": "hybrid_quantum_classical_survival_ensemble",
        "dataset_summary": summarize_dataset(dataset, train_idx=train_idx, test_idx=test_idx),
        "preprocessing": {
            "expression_transform": "log2(TPM + 1)",
            "clinical_features": ["age", "gender"] + SURVIVAL_CATEGORICAL_COLUMNS,
            "feature_selection": "Survival score ranking fitted on training data only",
            "scaling": "MinMaxScaler fitted on training data only",
            "categorical_encoding": "Training-fold one-hot encoding with rare-category collapse",
        },
        "validation_protocol": {
            "outer_split": "Stratified train/test split (80/20) on event indicator",
            "inner_cv": f"Stratified {INNER_CV_FOLDS}-fold cross-validation on the training partition",
            "selection_metric": "mean validation C-index",
            "survival_loss": "Cox partial log-likelihood plus pairwise ranking regularization",
        },
        "selected_hyperparameters": asdict(best_config),
        "train_cv_c_index": float(best_result["mean_val_c_index"]),
        "inner_cv_results_top5": cv_results.head(5).to_dict(orient="records"),
        "final_training_epochs": final_epochs,
        "ensemble_seeds": FINAL_ENSEMBLE_SEEDS,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_c_index_ci95": c_index_ci95,
        "risk_band_cutoffs": risk_band_cutoffs,
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
            "cv_results": cv_results_path.name,
            "holdout_predictions": holdout_predictions_path.name,
            "research_report": report_path.name,
            "explainability": explainability_path.name,
        },
        "notes": [
            "This model produces relative survival risk scores rather than exact predicted death dates.",
            "Final reported risk scores are the mean output of multiple independently trained hybrid survival models.",
            "Month-range outputs should be calibrated later from survival curves or risk-group summaries.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    explainability_path.write_text(json.dumps(explainability, indent=2), encoding="utf-8")
    write_markdown_report(metadata=metadata, cv_results=cv_results, report_path=report_path)

    print(f"Saved survival model state to {model_path}")
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
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
