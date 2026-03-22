import torch
import numpy as np
import shap
import sys; sys.path.insert(0, '.')
from backend.app.hybrid_model import HybridQuantumClassifier
from scripts.train_model import ModelConfig

# Create a dummy model
config = ModelConfig(
    n_top_genes=50, n_qubits=6, n_layers=2, hidden_dim=64,
    head_dim=32, temperature=0.5, entropy_lambda=0.0
)
model = HybridQuantumClassifier(config)
model.eval()

# Dummy data (batch of 10 for background, 2 for testing)
background = torch.randn(10, 52)
test_inputs = torch.randn(2, 52)

try:
    print("Trying GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_inputs)
    print("GradientExplainer SUCCESS!")
    print("Shape:", np.array(shap_values).shape)
except Exception as e:
    print("GradientExplainer FAILED:", e)

try:
    print("\nTrying DeepExplainer...")
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_inputs)
    print("DeepExplainer SUCCESS!")
except Exception as e:
    print("DeepExplainer FAILED:", e)
