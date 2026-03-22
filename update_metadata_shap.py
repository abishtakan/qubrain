import json, torch, numpy as np, shap, joblib, sys
sys.path.insert(0, '.')
from pathlib import Path
from backend.app.hybrid_model import HybridQuantumClassifier
from backend.app.explainability import build_global_explainability, shap_explain
from scripts.train_model import ARTIFACT_DIR

print('Loading metadata...')
with open(ARTIFACT_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print('Loading model state...')
checkpoint = torch.load(ARTIFACT_DIR / 'hybrid_model_state.pt', map_location='cpu', weights_only=True)
model_params = checkpoint['model_params']
model = HybridQuantumClassifier(**model_params)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print('Loading preprocessor and test data...')
preprocessor_dict = joblib.load(ARTIFACT_DIR / 'preprocessing.joblib')
preprocessor = preprocessor_dict['scaler']

with open(ARTIFACT_DIR / 'test_patients.json', 'r') as f:
    holdout_patients = json.load(f)

selected_genes = metadata['selected_genes']
X_test_unscaled = []
for p in holdout_patients:
    row = [p['age'], 1.0 if p['gender'] == 'male' else 0.0]
    row.extend([p['genes'][g] for g in selected_genes])
    X_test_unscaled.append(row)

X_test_unscaled = np.array(X_test_unscaled, dtype=np.float32)
X_test_scaled = preprocessor.transform(X_test_unscaled)
reference_scaled = np.median(X_test_scaled, axis=0, keepdims=True)

print('Computing SHAP values...')
holdout_attributions = shap_explain(
    model=model,
    inputs=X_test_scaled,
    baseline=reference_scaled,
    device='cpu',
)

feature_order = ['age', 'gender'] + selected_genes
explainability = build_global_explainability(
    feature_names=feature_order,
    attributions=holdout_attributions,
)

metadata['explainability'] = explainability

with open(ARTIFACT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print('SUCCESS METADATA UPDATED!')
