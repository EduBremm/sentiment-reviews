"""Serviço de predição — carrega modelo do disco e classifica textos."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> dict:
    path = path or settings.MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Rode `python -m src.models.train` antes."
        )
    logger.info("Carregando modelo de %s", path)
    return joblib.load(path)


def predict_one(text: str) -> dict:
    return predict_batch([text])[0]


def predict_batch(texts: list[str]) -> list[dict]:
    bundle = load_model()
    model = bundle["model"]
    labels = bundle["labels"]

    preds = model.predict(texts)
    out: list[dict] = []

    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(texts)
        for text, pred, p in zip(texts, preds, proba):
            idx = int(np.argmax(p))
            out.append({
                "text": text,
                "sentiment": labels[idx],
                "confidence": float(p[idx]),
                "probabilities": {lab: float(pv) for lab, pv in zip(labels, p)},
            })
    else:
        for text, pred in zip(texts, preds):
            out.append({
                "text": text,
                "sentiment": str(pred),
                "confidence": None,
                "probabilities": None,
            })

    return out
