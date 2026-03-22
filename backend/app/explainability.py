from __future__ import annotations

from typing import Any

import numpy as np
import shap
import torch

from .hybrid_model import HybridQuantumClassifier


def shap_explain(
    model: HybridQuantumClassifier,
    inputs: np.ndarray,
    baseline: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute feature contributions using shap.GradientExplainer, which computes 
    expected gradients combining ideas from Integrated Gradients and SHAP.
    """
    model.eval()

    input_array = np.asarray(inputs, dtype=np.float32)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    # GradientExplainer expects exactly the baseline distribution as a background dataset
    # We expand the single-vector baseline into a required format
    baseline_array = np.asarray(baseline, dtype=np.float32).reshape(1, -1)
    
    input_t = torch.from_numpy(input_array).to(device)
    baseline_t = torch.from_numpy(baseline_array).to(device)

    # SHAP internally expects the model output to be 2D: (batch_size, num_classes).
    # Our model returns a 1D tensor (batch_size,) for binary classification probability.
    # We use a simple wrapper to provide the expected 2D shape.
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            return self.base_model(x).unsqueeze(1)
            
    wrapped_model = ModelWrapper(model)

    # Initialize Explainer
    explainer = shap.GradientExplainer(wrapped_model, baseline_t)
    
    # explainer.shap_values returns a list of arrays (one for each output, though we have 1 risk output)
    # or a single numpy array depending on shap version.
    shap_vals = explainer.shap_values(input_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
        
    return np.asarray(shap_vals, dtype=np.float32)


def build_global_explainability(
    feature_names: list[str],
    attributions: np.ndarray,
    top_k: int | None = None,
) -> dict[str, Any]:
    attribution_array = np.asarray(attributions, dtype=np.float32)
    mean_abs = np.mean(np.abs(attribution_array), axis=0)
    mean_signed = np.mean(attribution_array, axis=0)

    rows = [
        {
            "feature": feature,
            "mean_absolute_attribution": float(mean_abs[index]),
            "mean_signed_attribution": float(mean_signed[index]),
        }
        for index, feature in enumerate(feature_names)
    ]
    rows.sort(key=lambda row: row["mean_absolute_attribution"], reverse=True)

    if top_k is not None:
        rows = rows[:top_k]

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return {
        "method": "Gradient SHAP",
        "aggregation": "Mean absolute SHAP value on the outer holdout cohort",
        "feature_importance": rows,
    }


def build_local_explanation(
    feature_names: list[str],
    patient_values: np.ndarray,
    reference_values: np.ndarray,
    attributions: np.ndarray,
    top_k: int = 5,
) -> dict[str, list[dict[str, float | str]]]:
    rows = []
    for feature, patient_value, reference_value, attribution in zip(
        feature_names,
        patient_values,
        reference_values,
        attributions,
    ):
        rows.append(
            {
                "feature": feature,
                "patient_value": float(patient_value),
                "reference_value": float(reference_value),
                "attribution": float(attribution),
                "absolute_attribution": float(abs(attribution)),
                "direction": "increases_risk" if attribution >= 0 else "reduces_risk",
            }
        )

    increasing = sorted(
        [row for row in rows if float(row["attribution"]) > 0],
        key=lambda row: float(row["absolute_attribution"]),
        reverse=True,
    )[:top_k]
    reducing = sorted(
        [row for row in rows if float(row["attribution"]) < 0],
        key=lambda row: float(row["absolute_attribution"]),
        reverse=True,
    )[:top_k]

    rows.sort(key=lambda row: float(row["absolute_attribution"]), reverse=True)
    return {
        "top_risk_increasing": increasing,
        "top_risk_reducing": reducing,
        "all_features": rows,
    }
