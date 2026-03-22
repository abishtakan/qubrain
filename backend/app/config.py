import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "hybrid_model_state.pt"
PREPROCESS_PATH = ARTIFACT_DIR / "preprocessing.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
SAMPLES_PATH = ARTIFACT_DIR / "test_patients.json"
EXPLAINABILITY_PATH = ARTIFACT_DIR / "explainability.json"
raw_origins = os.getenv(
    "QBRAIN_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
)
ALLOWED_ORIGINS = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]

# ---------------------------------------------------------------------------
# Demo credentials — OVERRIDE these via environment variables in any shared
# or production environment. The hardcoded fallback values here are for local
# development and demonstration only. See .env.example at the project root.
# ---------------------------------------------------------------------------
CLINICIAN_USERNAME = os.getenv("QBRAIN_UI_USERNAME", "oncologist")
CLINICIAN_PASSWORD = os.getenv("QBRAIN_UI_PASSWORD", "qubrain-demo-2026")
CLINICIAN_DISPLAY_NAME = os.getenv("QBRAIN_UI_DISPLAY_NAME", "Dr. QuBrain User")
AUTH_SESSION_HOURS = int(os.getenv("QBRAIN_UI_SESSION_HOURS", "12"))
