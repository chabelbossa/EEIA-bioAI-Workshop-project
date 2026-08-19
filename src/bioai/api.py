"""Optional FastAPI service for the compact student checkpoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import FEATURE_VERSION, advanced_features
from .model import StudentMLP, load_checkpoint, predict_probabilities


class PredictionRequest(BaseModel):
    sequence: str = Field(min_length=1, description="DNA sequence using A/C/G/T/N")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    probability_coding: float
    prediction: int
    threshold: float
    feature_version: str


@dataclass(slots=True)
class RuntimeState:
    model: StudentMLP | None = None
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


state = RuntimeState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    checkpoint = os.getenv("BIOAI_CHECKPOINT")
    if checkpoint:
        try:
            state.model, state.threshold, state.metadata = load_checkpoint(Path(checkpoint))
        except Exception as exc:  # surfaced by /health without hiding the root cause
            state.error = f"{type(exc).__name__}: {exc}"
    else:
        state.error = "BIOAI_CHECKPOINT is not configured"
    yield


app = FastAPI(
    title="EEIA bioAI student API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ready": state.model is not None,
        "error": state.error,
        "feature_version": FEATURE_VERSION,
        "checkpoint_metadata": state.metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail=state.error or "model unavailable")
    threshold = state.threshold if request.threshold is None else request.threshold
    vector = advanced_features(request.sequence)[None, :]
    probability = float(predict_probabilities(state.model, vector).reshape(-1)[0])
    return PredictionResponse(
        probability_coding=probability,
        prediction=int(probability >= threshold),
        threshold=threshold,
        feature_version=FEATURE_VERSION,
    )
