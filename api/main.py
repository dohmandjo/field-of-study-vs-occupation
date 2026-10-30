"""FastAPI service that scores a person with the saved PySpark pipeline.

    python -m ml.train                      # once, creates models/career_change_pipeline
    uvicorn api.main:app --port 8001        # from the repository root

Environment:
    MODEL_DIR     directory holding career_change_pipeline/ and feature_means.json (default: ./models)
    CORS_ORIGINS  comma-separated allowed origins (default: http://localhost:5173)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pyspark.ml import PipelineModel
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from ml.pipeline import coefficients_array, explain, get_spark, slot_names
from ml.schema import CATEGORICAL, FEATURES

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_SCHEMA = StructType(
    [StructField(f.model, StringType() if f.kind == "categorical" else DoubleType(), False) for f in FEATURES]
)


class PredictionRequest(BaseModel):
    """JSON contract used by the React form (camelCase field names)."""

    model_config = ConfigDict(extra="forbid")

    fieldOfStudy: str = Field(min_length=1)
    currentOccupation: str = Field(min_length=1)
    gender: str = Field(min_length=1)
    educationLevel: str = Field(min_length=1)
    industryGrowthRate: str = Field(min_length=1)
    familyInfluence: str = Field(min_length=1)
    age: int = Field(ge=16, le=100)
    yearsOfExperience: int = Field(ge=0, le=80)
    jobSatisfaction: int = Field(ge=1, le=10)
    workLifeBalance: int = Field(ge=1, le=10)
    jobOpportunities: int = Field(ge=0, le=100)
    salary: int = Field(ge=0)
    jobSecurity: int = Field(ge=1, le=10)
    careerChangeInterest: int = Field(ge=0, le=1)
    skillsGap: int = Field(ge=1, le=10)
    mentorshipAvailable: int = Field(ge=0, le=1)
    certifications: int = Field(ge=0, le=1)
    freelancingExperience: int = Field(ge=0, le=1)
    geographicMobility: int = Field(ge=0, le=1)
    professionalNetworks: int = Field(ge=1, le=10)
    careerChangeEvents: int = Field(ge=0, le=20)
    technologyAdoption: int = Field(ge=1, le=10)


class Effect(BaseModel):
    feature: str
    effect: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float = Field(description="P(likely to change occupation)")
    probability_0: float = Field(description="P(not likely to change occupation)")
    top_features: list[str]
    effects: list[Effect]
    unseen_categories: list[str] = Field(description="Category values the model never saw in training")


def load_model(model_dir: Path | None = None) -> dict:
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    spark = get_spark("career-change-api", master="local[1]")
    spark.sparkContext.setLogLevel("ERROR")
    pipeline_path = model_dir / "career_change_pipeline"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"{pipeline_path} not found - run `python -m ml.train` first")
    model = PipelineModel.load(str(pipeline_path))
    means = json.loads((model_dir / "feature_means.json").read_text())
    # categories seen in training, read from the fitted StringIndexers, to report unseen inputs
    known = {}
    for stage in model.stages:
        if hasattr(stage, "labelsArray") and stage.getInputCol() in {f.model for f in CATEGORICAL}:
            known[stage.getInputCol()] = set(stage.labelsArray[0])
    return {"spark": spark, "model": model, "means": means, "known": known}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_model()
    yield
    app.state.bundle["spark"].stop()


app = FastAPI(title="Career change predictor", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def score(bundle: dict, payload: PredictionRequest) -> PredictionResponse:
    spark, model = bundle["spark"], bundle["model"]
    row = {f.model: (str(getattr(payload, f.api)) if f.kind == "categorical" else float(getattr(payload, f.api)))
           for f in FEATURES}
    assembled = model.transform(spark.createDataFrame([row], _SCHEMA))
    out = assembled.select("features", "prediction", "probability").first()

    means = bundle["means"]
    names = slot_names(assembled)
    effects = explain(names, coefficients_array(model), means, out["features"].toArray())
    unseen = sorted(
        f"{f.api}={row[f.model]}" for f in CATEGORICAL if row[f.model] not in bundle["known"].get(f.model, ())
    )
    return PredictionResponse(
        prediction=int(out["prediction"]),
        probability=float(out["probability"][1]),
        probability_0=float(out["probability"][0]),
        top_features=[e["feature"] for e in effects],
        effects=effects,
        unseen_categories=unseen,
    )


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "bundle")}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return score(bundle, payload)
