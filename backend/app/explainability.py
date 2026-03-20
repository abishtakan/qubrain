from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .hybrid_model import HybridQuantumClassifier


def integrated_gradients(
    model: HybridQuantumClassifier,
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

    attributions: list[np.ndarray] = []
    for row in input_array:
        input_t = torch.from_numpy(row).to(device)
        total_grads = torch.zeros_like(input_t)
        delta = input_t - baseline_t

        for alpha in alphas:
            model.zero_grad(set_to_none=True)
            interpolated = (baseline_t + (alpha * delta)).unsqueeze(0).clone().detach().requires_grad_(True)
            output = model(interpolated)
            output.backward(torch.ones_like(output))
            total_grads += interpolated.grad.detach()[0]

        average_grads = total_grads / float(len(alphas))
        attribution = (delta * average_grads).detach().cpu().numpy().astype(np.float32)
        attributions.append(attribution)

    return np.vstack(attributions)


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
        "method": "Integrated Gradients",
        "aggregation": "Mean absolute attribution on the outer holdout cohort",
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
