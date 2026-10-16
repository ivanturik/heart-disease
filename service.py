"""API для модели и метаданных из notebooks/heart.ipynb."""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_DIR = Path(__file__).resolve().parent / "models"
NOTICE = (
    "Учебная оценка не подтверждает и не исключает заболевание. "
    "Score не является персональной вероятностью заболевания. "
    "Модель не определяет срочность медицинской помощи."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not all((MODEL_DIR / name).is_file() for name in ("model.joblib", "meta.json")):
        raise RuntimeError("Выполните notebooks/heart.ipynb: модель и meta.json должны быть в models/.")
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    required = {"model", "features", "threshold", "medians", "modes", "unknown_columns",
                "unknown_value", "categorical_columns", "test_metrics"}
    if (meta.get("schema_version") != 2 or meta.get("input_age_unit") != "years"
            or not required.issubset(meta) or not 0 <= meta["threshold"] <= 1):
        raise RuntimeError("Метаданные устарели. Повторите сохранение из notebooks/heart.ipynb.")
    app.state.meta = meta
    app.state.model = joblib.load(MODEL_DIR / "model.joblib")
    yield


app = FastAPI(
    title="Heart Disease Scoring", version="1.0.0", lifespan=lifespan,
    description="Оценка сочетания признаков и пояснение результата учебной модели. " + NOTICE,
)


class HeartInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, json_schema_extra={"example": {
        "age": 55, "gender": 1, "height": 165, "weight": 72, "ap_hi": 125, "ap_lo": 80,
        "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": None, "active": 1}})
    age: float = Field(ge=20, lt=80, description="Возраст в годах: от 20 до 79 полных лет")
    gender: Literal[1, 2] = Field(description="1 — женский, 2 — мужской")
    height: float | None = Field(default=None, ge=120, le=220, description="Рост, см")
    weight: float | None = Field(default=None, ge=30, le=200, description="Вес, кг")
    ap_hi: float | None = Field(default=None, ge=70, le=250, description="Верхнее давление, мм рт. ст.")
    ap_lo: float | None = Field(default=None, ge=40, le=150, description="Нижнее давление, мм рт. ст.")
    cholesterol: Literal[1, 2, 3] | None = Field(default=None, description="Общий холестерин: 1 — <200, 2 — 200–<240, 3 — ≥240 мг/дл")
    gluc: Literal[1, 2, 3] | None = Field(default=None, description="HbA1c: 1 — <5.7%, 2 — 5.7–<6.5%, 3 — ≥6.5%")
    smoke: Literal[0, 1] | None = Field(default=None, description="Курение: 0 — нет, 1 — да")
    alco: Literal[0, 1] | None = Field(default=None, description="Употребление алкоголя: 0 — нет, 1 — да")
    active: Literal[0, 1] | None = Field(default=None, description="Физическая активность: 0 — нет, 1 — да")

    @model_validator(mode="after")
    def check_pairs(self):
        if (self.ap_hi is None) != (self.ap_lo is None):
            raise ValueError("Укажите оба значения давления или ни одного.")
        if (self.height is None) != (self.weight is None):
            raise ValueError("Укажите рост и вес вместе или оставьте оба поля пустыми.")
        if self.ap_hi is not None and self.ap_lo is not None and self.ap_hi <= self.ap_lo:
            raise ValueError("Верхнее давление должно быть выше нижнего.")
        return self


class HeartResult(BaseModel):
    model: str
    score: float = Field(ge=0, le=1, description="Оценка класса cardio=1, не клинический риск")
    threshold: float = Field(ge=0, le=1)
    above_threshold: bool
    recommendation: str
    imputed_fields: list[str] = Field(description="Неуказанные поля, обработанные по правилам обучения")
    test_metrics: dict[str, float] = Field(description="Метрики на test этого набора данных")
    notice: str


def prepare_features(values: dict, meta: dict) -> pd.DataFrame:
    frame = pd.DataFrame([values], dtype=float)
    frame = frame.fillna(meta["medians"]).fillna(meta["modes"])
    unknown = meta["unknown_columns"]
    frame[unknown] = frame[unknown].fillna(meta["unknown_value"])
    categories = meta["categorical_columns"]
    frame[categories] = frame[categories].astype(int).astype(str)
    frame["bmi"] = frame["weight"] / (frame["height"] / 100) ** 2
    frame["pulse_pressure"] = frame["ap_hi"] - frame["ap_lo"]
    return frame[meta["features"]]


@app.get("/")
@app.get("/health")
def health(request: Request):
    return {"ok": True, "model": request.app.state.meta["model"], "docs": "/docs"}


@app.post("/predict", response_model=HeartResult)
def predict(client: HeartInput, request: Request):
    values = client.model_dump()
    meta = request.app.state.meta
    frame = prepare_features(values, meta)
    score = float(request.app.state.model.predict_proba(frame)[0, 1])
    above_threshold = score >= meta["threshold"]
    recommendation = (
        "Стоит обсудить введённые показатели с врачом. "
        "Оценка выше порога модели сама по себе не означает наличие заболевания."
        if above_threshold else
        "При жалобах обратитесь к врачу независимо от этой оценки. "
        "Результат ниже порога не является подтверждением здоровья."
    )
    return HeartResult(
        model=meta["model"], score=score, threshold=meta["threshold"],
        above_threshold=above_threshold, recommendation=recommendation,
        imputed_fields=[name for name, value in values.items() if value is None],
        test_metrics=meta["test_metrics"], notice=NOTICE,
    )
