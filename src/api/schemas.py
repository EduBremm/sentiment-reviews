"""Schemas Pydantic da API."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="O app trava toda hora, péssima experiência")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: Optional[float]
    probabilities: Optional[dict[str, float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_ok: bool
    timestamp: datetime


class SentimentStats(BaseModel):
    sentiment: str
    count: int


class VersionStats(BaseModel):
    app_version: str
    positive: int
    neutral: int
    negative: int
    total: int


class TimelinePoint(BaseModel):
    date: str
    positive: int
    neutral: int
    negative: int


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    top_terms: list[str]
    sample_reviews: list[str]
