"""Scoring API."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path(os.environ.get("ABUSE_DETECT_ARTIFACTS", "artifacts"))
MAX_CHARS = 5000


class ScoreRequest(BaseModel):
    subject: str = Field("", max_length=1000)
    body: str = Field(..., max_length=200_000)


class ScoreResponse(BaseModel):
    spam_probability: float
    decision: str
    threshold: float
    model_version: str


@lru_cache(maxsize=1)
def _load():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    return model, meta


app = FastAPI(title="abuse-detect", version="1.0.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    _, meta = _load()
    return {"status": "ok", "model_version": meta["model_version"]}


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    model, meta = _load()
    text = (req.subject + "\n" + req.body)[:MAX_CHARS]
    p = float(model.predict_proba([text])[0, 1])
    return ScoreResponse(
        spam_probability=round(p, 6),
        decision="block" if p >= meta["threshold"] else "allow",
        threshold=meta["threshold"],
        model_version=meta["model_version"],
    )
