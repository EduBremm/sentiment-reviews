"""Clusterização de reclamações (K-Means sobre reviews NEGATIVOS).

Descobre temas recorrentes de insatisfação (bugs, cobranças, propaganda, ...).

Uso:
    python -m src.models.clustering --k 5
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import MODELS_DIR, settings
from src.data.preprocess import preprocess
from src.db.database import get_engine, get_session_factory
from src.db.models import ClusterAssignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_negative_reviews() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT review_id, text FROM reviews WHERE sentiment = 'NEGATIVO' AND text IS NOT NULL",
            conn,
        )
    logger.info("Carregados %s reviews negativos", len(df))
    return df


def top_terms_per_cluster(vec: TfidfVectorizer, km: KMeans, n: int = 8) -> dict[int, list[str]]:
    terms = np.array(vec.get_feature_names_out())
    order = km.cluster_centers_.argsort()[:, ::-1]
    return {i: terms[order[i, :n]].tolist() for i in range(km.n_clusters)}


def find_best_k(X, k_min: int = 3, k_max: int = 8) -> int:
    """Escolhe k com melhor Silhouette (usando amostra se dataset grande)."""
    sample_idx = np.random.RandomState(settings.RANDOM_SEED).choice(
        X.shape[0], size=min(2000, X.shape[0]), replace=False,
    )
    Xs = X[sample_idx]

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init="auto")
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xs, labels)
        logger.info("k=%s  silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def run_clustering(k: int | None = None, out_dir: Path = MODELS_DIR) -> dict:
    df = load_negative_reviews()
    if len(df) < 30:
        raise RuntimeError(f"Muito poucos reviews negativos ({len(df)}) para clusterizar.")

    vec = TfidfVectorizer(
        preprocessor=preprocess,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=5000,
    )
    X = vec.fit_transform(df["text"].values)

    if k is None:
        k = find_best_k(X)
        logger.info("Melhor k encontrado: %s", k)

    km = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init="auto")
    labels = km.fit_predict(X)

    silh = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    db = davies_bouldin_score(X.toarray(), labels) if len(set(labels)) > 1 else None
    top = top_terms_per_cluster(vec, km)

    logger.info("Silhouette=%.4f  DaviesBouldin=%.4f", silh or -1, db or -1)
    for cid, terms in top.items():
        logger.info("Cluster %s: %s", cid, ", ".join(terms))

    # Persiste
    _save_assignments(df, labels, top)

    bundle_path = out_dir / "clustering.pkl"
    joblib.dump({"vectorizer": vec, "kmeans": km, "top_terms": top}, bundle_path)

    metrics_path = out_dir / "clustering_metrics.json"
    payload = {
        "trained_at": datetime.utcnow().isoformat(),
        "k": k,
        "n_reviews": len(df),
        "silhouette": float(silh) if silh is not None else None,
        "davies_bouldin": float(db) if db is not None else None,
        "top_terms": top,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _save_assignments(df: pd.DataFrame, labels, top_terms: dict[int, list[str]]) -> None:
    SessionFactory = get_session_factory()
    with SessionFactory() as session:
        session.query(ClusterAssignment).delete()
        rows = [
            ClusterAssignment(
                review_id=rid,
                cluster_id=int(lbl),
                top_terms=", ".join(top_terms[int(lbl)]),
            )
            for rid, lbl in zip(df["review_id"].values, labels)
        ]
        session.add_all(rows)
        session.commit()
    logger.info("Assignments salvos no banco (%s)", len(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="k (padrão: seleciona por silhouette)")
    args = parser.parse_args()
    run_clustering(k=args.k)


if __name__ == "__main__":
    main()
