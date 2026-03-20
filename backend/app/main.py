from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .auth import require_current_user, session_manager
from .config import ALLOWED_ORIGINS
from .predictor import get_predictor
from .schemas import (
    CurrentUserResponse,
    ExplanationRequest,
    ExplanationResponse,
    LoginRequest,
    LoginResponse,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(
    title="QuBrain Mortality Classifier API",
    version="2.0.0",
    description="QuBrain hybrid quantum-classical GBM mortality classification API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    predictor = get_predictor()
    return {
        "status": "ok",
        "model": predictor.get_metadata()["selected_model"],
        "task": predictor.get_metadata()["task"],
    }


@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    return LoginResponse(**session_manager.login(request.username, request.password))


@app.get("/auth/me", response_model=CurrentUserResponse)
def current_user(user: dict = Depends(require_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse(**user)


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    session_manager.logout(authorization)
    return {"status": "signed_out"}


@app.get("/metadata")
def metadata(user: dict = Depends(require_current_user)) -> dict:
    return get_predictor().get_metadata()


@app.get("/samples/random")
def sample_patient(user: dict = Depends(require_current_user)) -> dict:
    try:
        return get_predictor().get_random_test_patient()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, user: dict = Depends(require_current_user)) -> PredictionResponse:
    try:
        result = get_predictor().predict(
            age=request.age,
            gender=request.gender,
            genes=request.genes,
        )
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: ExplanationRequest, user: dict = Depends(require_current_user)) -> ExplanationResponse:
    try:
        result = get_predictor().explain(
            age=request.age,
            gender=request.gender,
            genes=request.genes,
        )
        return ExplanationResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
