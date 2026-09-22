"""API FastAPI — App Review Intelligence."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import case, func, text
from sqlalchemy.orm import Session

from src.api.schemas import (
    ClusterInfo,
    HealthResponse,
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
    SentimentStats,
    TimelinePoint,
    VersionStats,
)
from src.db.database import get_engine, get_session_factory, init_db
from src.db.models import ClusterAssignment, Prediction, Review
from src.models.predict import load_model, predict_batch, predict_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="App Review Intelligence API",
    description="Classificação de sentimento e análise de reviews de aplicativos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


SessionLocal = get_session_factory()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def _startup() -> None:
    try:
        init_db()
        logger.info("DB inicializado.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Falha ao inicializar DB: %s", exc)


# ============================================================
# Health
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    model_ok = True
    try:
        load_model()
    except Exception:
        model_ok = False

    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok" if model_ok and db_ok else "degraded",
        model_loaded=model_ok,
        db_ok=db_ok,
        timestamp=datetime.utcnow(),
    )


# ============================================================
# Predição
# ============================================================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    try:
        result = predict_one(req.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Log a predição para auditoria
    try:
        db.add(Prediction(
            text=req.text,
            predicted_sentiment=result["sentiment"],
            confidence=result.get("confidence"),
            model_name=load_model().get("name"),
        ))
        db.commit()
    except Exception as exc:  # pragma: no cover
        logger.warning("Falha registrando predição: %s", exc)
        db.rollback()

    return PredictResponse(**result)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_many(req: PredictBatchRequest) -> list[PredictResponse]:
    try:
        results = predict_batch(req.texts)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [PredictResponse(**r) for r in results]


# ============================================================
# Estatísticas
# ============================================================
@app.get("/reviews/stats", response_model=list[SentimentStats])
def stats(app_id: Optional[str] = None, db: Session = Depends(get_db)) -> list[SentimentStats]:
    q = db.query(Review.sentiment, func.count(Review.review_id))
    if app_id:
        q = q.filter(Review.app_id == app_id)
    q = q.group_by(Review.sentiment)
    return [SentimentStats(sentiment=s or "?", count=c) for s, c in q.all()]


@app.get("/reviews/by-version", response_model=list[VersionStats])
def by_version(app_id: Optional[str] = None, db: Session = Depends(get_db)) -> list[VersionStats]:
    q = db.query(
        Review.app_version,
        func.sum(case((Review.sentiment == "POSITIVO", 1), else_=0)).label("positive"),
        func.sum(case((Review.sentiment == "NEUTRO", 1), else_=0)).label("neutral"),
        func.sum(case((Review.sentiment == "NEGATIVO", 1), else_=0)).label("negative"),
        func.count(Review.review_id).label("total"),
    )
    if app_id:
        q = q.filter(Review.app_id == app_id)
    q = q.filter(Review.app_version.isnot(None)).group_by(Review.app_version).order_by(Review.app_version.desc())

    return [
        VersionStats(
            app_version=v or "?",
            positive=int(p or 0),
            neutral=int(n or 0),
            negative=int(neg or 0),
            total=int(t or 0),
        )
        for v, p, n, neg, t in q.limit(20).all()
    ]


@app.get("/reviews/timeline", response_model=list[TimelinePoint])
def timeline(
    app_id: Optional[str] = None,
    granularity: str = Query("day", pattern="^(day|week|month)$"),
    db: Session = Depends(get_db),
) -> list[TimelinePoint]:
    trunc = {"day": "day", "week": "week", "month": "month"}[granularity]
    sql = text(
        f"""
        SELECT
            date_trunc(:trunc, created_at)::date AS d,
            SUM(CASE WHEN sentiment = 'POSITIVO' THEN 1 ELSE 0 END) AS pos,
            SUM(CASE WHEN sentiment = 'NEUTRO'   THEN 1 ELSE 0 END) AS neu,
            SUM(CASE WHEN sentiment = 'NEGATIVO' THEN 1 ELSE 0 END) AS neg
        FROM reviews
        WHERE (:app_id IS NULL OR app_id = :app_id) AND created_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )
    rows = db.execute(sql, {"trunc": trunc, "app_id": app_id}).fetchall()
    return [
        TimelinePoint(date=str(r[0]), positive=int(r[1]), neutral=int(r[2]), negative=int(r[3]))
        for r in rows
    ]


@app.get("/reviews/clusters", response_model=list[ClusterInfo])
def clusters(db: Session = Depends(get_db)) -> list[ClusterInfo]:
    sql = text(
        """
        SELECT ca.cluster_id, ca.top_terms, COUNT(*) AS n
        FROM cluster_assignments ca
        GROUP BY ca.cluster_id, ca.top_terms
        ORDER BY ca.cluster_id
        """
    )
    grouped = db.execute(sql).fetchall()

    out: list[ClusterInfo] = []
    for cid, terms, n in grouped:
        samples_sql = text(
            """
            SELECT r.text
            FROM cluster_assignments ca
            JOIN reviews r ON r.review_id = ca.review_id
            WHERE ca.cluster_id = :cid
            LIMIT 3
            """
        )
        samples = [row[0] for row in db.execute(samples_sql, {"cid": cid}).fetchall()]
        out.append(ClusterInfo(
            cluster_id=int(cid),
            size=int(n),
            top_terms=[t.strip() for t in (terms or "").split(",") if t.strip()],
            sample_reviews=samples,
        ))
    return out
