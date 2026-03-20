from __future__ import annotations

import json
import random
from functools import lru_cache

import joblib
import numpy as np
import torch

from .config import EXPLAINABILITY_PATH, METADATA_PATH, MODEL_PATH, PREPROCESS_PATH, SAMPLES_PATH
from .explainability import build_local_explanation, integrated_gradients
from .hybrid_model import HybridQuantumClassifier


class ProductionPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        if not PREPROCESS_PATH.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found: {PREPROCESS_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata artifact not found: {METADATA_PATH}")

        model_payload = torch.load(MODEL_PATH, map_location="cpu")
        preprocess_payload = joblib.load(PREPROCESS_PATH)

        model_params = model_payload.get("model_params", {"n_features": model_payload["n_features"]})
        self.model = HybridQuantumClassifier(**model_params)
        self.model.load_state_dict(model_payload["state_dict"])
        self.model.eval()
        self.scaler = preprocess_payload["scaler"]
        self.metadata = json.loads(METADATA_PATH.read_text())
        self.metadata.setdefault("decision_threshold", 0.5)
        if EXPLAINABILITY_PATH.exists():
            self.explainability = json.loads(EXPLAINABILITY_PATH.read_text())
        else:
            self.explainability = {"feature_importance": []}
        self.test_patients = json.loads(SAMPLES_PATH.read_text()) if SAMPLES_PATH.exists() else []
        self.selected_genes: list[str] = preprocess_payload["selected_genes"]
        self.feature_order: list[str] = preprocess_payload.get("feature_order", ["age", "gender"] + self.selected_genes)
        self.reference_unscaled = np.asarray(
            preprocess_payload.get("reference_unscaled", np.zeros(len(self.feature_order), dtype=np.float32)),
            dtype=np.float32,
        )
        self.reference_scaled = np.asarray(
            preprocess_payload.get("reference_scaled", np.zeros(len(self.feature_order), dtype=np.float32)),
            dtype=np.float32,
        )
        self.decision_threshold = float(self.metadata.get("decision_threshold", 0.5))

    def get_metadata(self) -> dict:
        return self.metadata

    def get_random_test_patient(self) -> dict:
        if not self.test_patients:
            raise ValueError("No saved test patients were found.")
        return random.choice(self.test_patients)

    def _vectorize(self, age: float, gender: str, genes: dict[str, float]) -> np.ndarray:
        missing = [gene for gene in self.selected_genes if gene not in genes]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing {len(missing)} required genes. First missing genes: {preview}")

        extra = [gene for gene in genes if gene not in self.selected_genes]
        if extra:
            preview = ", ".join(extra[:5])
            raise ValueError(f"Received {len(extra)} unexpected genes. First unexpected genes: {preview}")

        gender_value = 1 if gender == "male" else 0
        ordered_genes = [float(genes[gene]) for gene in self.selected_genes]
        return np.array([[float(age), gender_value] + ordered_genes], dtype=float)

    def predict(self, age: float, gender: str, genes: dict[str, float]) -> dict:
        vector = self._vectorize(age, gender, genes)
        scaled = self.scaler.transform(vector).astype(np.float32)
        with torch.no_grad():
            mortality_probability = float(self.model(torch.from_numpy(scaled)).item())
        alive_probability = float(1.0 - mortality_probability)

        return {
            "prediction": "Dead" if mortality_probability >= self.decision_threshold else "Alive",
            "mortality_probability": mortality_probability,
            "alive_probability": alive_probability,
            "model_name": self.metadata["selected_model"],
            "decision_threshold": self.decision_threshold,
        }

    def explain(self, age: float, gender: str, genes: dict[str, float]) -> dict:
        vector = self._vectorize(age, gender, genes)
        scaled = self.scaler.transform(vector).astype(np.float32)

        with torch.no_grad():
            mortality_probability = float(self.model(torch.from_numpy(scaled)).item())

        attributions = integrated_gradients(
            model=self.model,
            inputs=scaled,
            baseline=self.reference_scaled,
            steps=24,
            device="cpu",
        )[0]
        local = build_local_explanation(
            feature_names=self.feature_order,
            patient_values=vector[0],
            reference_values=self.reference_unscaled,
            attributions=attributions,
            top_k=5,
        )

        return {
            "method": "Integrated Gradients",
            "baseline_description": "Attributions are computed relative to the median feature profile of the training cohort.",
            "prediction": "Dead" if mortality_probability >= self.decision_threshold else "Alive",
            "mortality_probability": mortality_probability,
            "decision_threshold": self.decision_threshold,
            "top_risk_increasing": local["top_risk_increasing"],
            "top_risk_reducing": local["top_risk_reducing"],
            "global_top_features": self.explainability.get("feature_importance", [])[:8],
        }


@lru_cache(maxsize=1)
def get_predictor() -> ProductionPredictor:
    return ProductionPredictor()
