"""Treina e compara Naive Bayes, Regressão Logística e SVM.

Uso:
    python -m src.models.train
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, settings
from src.data.preprocess import preprocess
from src.db.database import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["NEGATIVO", "NEUTRO", "POSITIVO"]


@dataclass
class ModelResult:
    name: str
    accuracy: float
    f1_macro: float
    roc_auc_ovr: float | None
    report: dict
    confusion: list[list[int]]


def load_data_from_db() -> pd.DataFrame:
    """Lê os reviews rotulados da tabela ``reviews``."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT text, sentiment FROM reviews WHERE text IS NOT NULL", conn)
    logger.info("Carregados %s reviews do banco", len(df))
    return df


def load_data_from_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["text", "sentiment"]].dropna()


def build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=20_000,
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])


def _get_proba(pipe: Pipeline, X):
    """Retorna probabilidades por classe. LinearSVC é calibrado dentro do pipeline."""
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        return pipe.predict_proba(X)
    return None


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> ModelResult:
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", labels=LABELS)
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS).tolist()

    roc_auc = None
    proba = _get_proba(pipe, X_test)
    if proba is not None:
        lb = LabelBinarizer().fit(LABELS)
        y_bin = lb.transform(y_test)
        try:
            roc_auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = None

    logger.info(
        "%s | acc=%.4f  f1_macro=%.4f  roc_auc=%s",
        name, acc, f1, f"{roc_auc:.4f}" if roc_auc else "n/a",
    )
    return ModelResult(name=name, accuracy=acc, f1_macro=f1, roc_auc_ovr=roc_auc, report=report, confusion=cm)


def train_all(df: pd.DataFrame, out_dir: Path = MODELS_DIR) -> dict:
    """Treina, compara e salva o melhor modelo (por F1 macro)."""
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df = df[df["sentiment"].isin(LABELS)]
    logger.info("Distribuição de classes:\n%s", df["sentiment"].value_counts().to_string())

    X, y = df["text"].values, df["sentiment"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=settings.RANDOM_SEED,
    )

    candidates = {
        "naive_bayes": build_pipeline(MultinomialNB()),
        "logistic_regression": build_pipeline(LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=settings.RANDOM_SEED,
        )),
        "svm_linear": build_pipeline(CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=settings.RANDOM_SEED),
            cv=3,
        )),
    }

    results: list[ModelResult] = []
    fitted: dict[str, Pipeline] = {}

    for name, pipe in candidates.items():
        logger.info("Treinando %s ...", name)
        pipe.fit(X_train, y_train)
        results.append(evaluate(name, pipe, X_test, y_test))
        fitted[name] = pipe

    best = max(results, key=lambda r: r.f1_macro)
    logger.info("🏆 Melhor modelo: %s (F1 macro = %.4f)", best.name, best.f1_macro)

    best_path = out_dir / "best_model.pkl"
    joblib.dump({"model": fitted[best.name], "name": best.name, "labels": LABELS}, best_path)
    logger.info("Modelo salvo em %s", best_path)

    metrics_path = out_dir / "metrics.json"
    payload = {
        "trained_at": datetime.utcnow().isoformat(),
        "best_model": best.name,
        "n_samples": len(df),
        "class_distribution": df["sentiment"].value_counts().to_dict(),
        "results": [r.__dict__ for r in results],
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    logger.info("Métricas salvas em %s", metrics_path)

    return payload


def main() -> None:
    try:
        df = load_data_from_db()
    except Exception as exc:  # pragma: no cover
        logger.warning("Falha lendo do banco (%s). Tentando CSV mais recente...", exc)
        from src.config import RAW_DIR
        csvs = sorted(RAW_DIR.glob("reviews_*.csv"))
        if not csvs:
            raise RuntimeError("Sem dados para treinar. Rode src.data.collect antes.") from exc
        df = load_data_from_csv(csvs[-1])

    if len(df) < 100:
        raise RuntimeError(
            f"Só {len(df)} reviews disponíveis — colete mais antes de treinar (mínimo ~500)."
        )

    train_all(df)


if __name__ == "__main__":
    main()
