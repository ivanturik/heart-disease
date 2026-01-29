from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "meta.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Артефакты не найдены. Сначала запусти: python train.py")

    app.state.model = joblib.load(MODEL_PATH)
    app.state.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    app.state.features = app.state.meta["features"]
    app.state.threshold = float(app.state.meta.get("threshold", 0.5))

    yield


app = FastAPI(title="Heart Disease Scoring API", lifespan=lifespan)


class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def root(request: Request):
    model = getattr(request.app.state, "model", None)
    features = getattr(request.app.state, "features", None)
    threshold = getattr(request.app.state, "threshold", 0.5)

    return {
        "ok": True,
        "docs": "/docs",
        "model_loaded": model is not None,
        "n_features": len(features) if features else 0,
        "threshold": threshold,
    }


@app.post("/predict")
def predict(x: HeartInput, request: Request):
    model = getattr(request.app.state, "model", None)
    features = getattr(request.app.state, "features", None)
    threshold = getattr(request.app.state, "threshold", 0.5)

    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = x.model_dump() if hasattr(x, "model_dump") else x.dict()
    X = pd.DataFrame([data], columns=features)

    p = float(model.predict_proba(X)[:, 1][0])
    pred = int(p >= threshold)

    return {
        "p_target": p,
        "threshold": threshold,
        "predicted_target": pred,
    }
