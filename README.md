# QuBrain Research Package

This folder is the cleaned research package for the hybrid quantum-classical GBM project. It keeps the deployable backend and frontend, but the training pipeline is now designed to be academically defensible as well as runnable.

## What Changed

Compared with the earlier simplified package, the training workflow now includes:

- stratified outer holdout evaluation
- inner cross-validation for hyperparameter selection
- fold-local feature selection and scaling
- leak-free class balancing inside training folds only
- entropy-regularized hybrid training
- validation-based threshold selection instead of assuming `0.5`
- overfitting analysis
- integrated-gradients explainability for global and patient-level interpretation
- saved research artifacts such as `cv_results.csv`, `holdout_predictions.csv`, and `research_report.md`

Important scope note:

- This package performs **mortality-status classification** (`Alive` vs `Dead`) from TCGA clinical status.
- It does **not** claim time-to-event survival analysis.

## Folder Layout

```text
qubrain/
  scripts/
    train_model.py
    start_backend.ps1
    start_frontend.ps1
  backend/
    app/
    model_artifacts/
    tests/
  frontend/
  RESEARCH_PLAYBOOK.md
```

## 1. Install Requirements

From the repository root:

```bash
pip install -r qubrain/requirements.txt
```

## 2. Train The Research Model

Full research search:

```bash
python qubrain/scripts/train_model.py
```

Faster smoke-test search:

```bash
python qubrain/scripts/train_model.py --quick
```

This writes:

- `qubrain/backend/model_artifacts/hybrid_model_state.pt`
- `qubrain/backend/model_artifacts/preprocessing.joblib`
- `qubrain/backend/model_artifacts/metadata.json`
- `qubrain/backend/model_artifacts/test_patients.json`
- `qubrain/backend/model_artifacts/cv_results.csv`
- `qubrain/backend/model_artifacts/holdout_predictions.csv`
- `qubrain/backend/model_artifacts/research_report.md`
- `qubrain/backend/model_artifacts/explainability.json`

## 3. Benchmark Classical Baselines

```bash
python qubrain/scripts/benchmark_baselines.py
```

This writes:

- `qubrain/backend/model_artifacts/baseline_benchmark.csv`
- `qubrain/backend/model_artifacts/baseline_benchmark.md`

## 4. Start The Backend

Use a normal foreground terminal. No background launcher is required.

Direct command:

```bash
python -m uvicorn qubrain.backend.app.main:app --host 127.0.0.1 --port 8010 --reload
```

Useful endpoints:

- `GET /health`
- `GET /metadata`
- `GET /samples/random`
- `POST /predict`
- `POST /explain`

## 5. Start The Frontend

Use a second normal foreground terminal.

Direct commands:

```bash
cd qubrain/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5174
```

The frontend expects the backend on port `8010` and runs on `5174`.

## 6. Run Backend Smoke Tests

From the repository root:

```bash
python -m unittest discover qubrain/backend/tests
```

## 7. Study Notes For The Thesis

For the current validated methodology and metrics, use:

- `qubrain/THESIS_METHOD_ARTIFACT.md`

For viva-style explanation notes and speaking points, use:

- `qubrain/RESEARCH_PLAYBOOK.md`
