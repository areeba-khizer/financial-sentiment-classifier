"""FastAPI service exposing the fine-tuned financial sentiment model.

Run locally:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import get_classifier

MAX_BATCH = 256


class PredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
        description="One or more pieces of financial news text to classify.",
        examples=[["The company beat earnings expectations and raised guidance."]],
    )


class Prediction(BaseModel):
    text: str
    label: str
    score: float
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model on startup so the first request isn't slow. If no model is
    # present yet, defer the error to request time rather than crashing boot.
    try:
        get_classifier()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Financial News Sentiment Classifier",
    description="Fine-tuned DistilBERT classifying financial news as positive, negative, or neutral.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    try:
        get_classifier()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/labels")
def labels() -> dict:
    classifier = _require_model()
    return {"labels": list(classifier.id2label.values())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    classifier = _require_model()
    outputs = classifier.predict(request.texts)
    predictions = [
        Prediction(text=text, **output)
        for text, output in zip(request.texts, outputs)
    ]
    return PredictResponse(predictions=predictions)


def _require_model():
    try:
        return get_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train it with `python -m src.train`.",
        ) from exc
