from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/heart.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "meta.json"

TARGET_COL = "target"
RANDOM_STATE = 42

TARGET_RECALL = 0.90

RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def pick_threshold_for_recall(y_true: np.ndarray, prob: np.ndarray, target_recall: float) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    thr = np.append(thr, 1.0)

    mask = rec >= target_recall
    if mask.any():
        return float(thr[mask][-1])
    return 0.5


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"В данных нет колонки '{TARGET_COL}'")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold_for_recall(y_val.to_numpy(), val_prob, TARGET_RECALL)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    }

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    model.fit(X_train_full, y_train_full)

    joblib.dump(model, MODEL_PATH)

    meta = {
        "model": "RandomForestClassifier",
        "features": list(X.columns),
        "threshold": float(threshold),
        "target_recall_goal": float(TARGET_RECALL),
        "rf_params": RF_PARAMS,
        "metrics_test": metrics,
        "data_path": str(DATA_PATH),
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta : {META_PATH}")
    print("TEST metrics:", metrics)
    print("Chosen threshold:", round(threshold, 6))


if __name__ == "__main__":
    main()
