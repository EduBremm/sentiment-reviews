"""Modelos ORM (tabelas)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Review(Base):
    __tablename__ = "reviews"

    review_id = Column(String(128), primary_key=True)
    app_id = Column(String(128), index=True, nullable=False)
    user_name = Column(String(255))
    text = Column(Text, nullable=False)
    rating = Column(Integer)
    sentiment = Column(String(16), index=True)  # POSITIVO / NEUTRO / NEGATIVO (rótulo fraco)
    app_version = Column(String(64), index=True)
    thumbs_up = Column(Integer, default=0)
    created_at = Column(DateTime, index=True)
    collected_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(128), index=True)
    text = Column(Text, nullable=False)
    predicted_sentiment = Column(String(16), nullable=False)
    confidence = Column(Float)
    model_name = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)


class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(128), index=True)
    cluster_id = Column(Integer, index=True)
    top_terms = Column(Text)  # separados por vírgula
    created_at = Column(DateTime, default=datetime.utcnow)
