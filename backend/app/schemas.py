from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gender: Literal["male", "female"]
    genes: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: Literal["Alive", "Dead"]
    mortality_probability: float
    alive_probability: float
    model_name: str
    decision_threshold: float


class ExplanationRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gender: Literal["male", "female"]
    genes: dict[str, float]


class FeatureContribution(BaseModel):
    feature: str
    patient_value: float
    reference_value: float
    attribution: float
    absolute_attribution: float
    direction: Literal["increases_risk", "reduces_risk"]


class GlobalFeatureImportance(BaseModel):
    feature: str
    mean_absolute_attribution: float
    mean_signed_attribution: float
    rank: int


class ExplanationResponse(BaseModel):
    method: str
    baseline_description: str
    prediction: Literal["Alive", "Dead"]
    mortality_probability: float
    decision_threshold: float
    top_risk_increasing: list[FeatureContribution]
    top_risk_reducing: list[FeatureContribution]
    global_top_features: list[GlobalFeatureImportance]


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"]
    username: str
    display_name: str
    expires_in_hours: int


class CurrentUserResponse(BaseModel):
    username: str
    display_name: str
