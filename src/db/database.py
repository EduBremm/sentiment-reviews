"""Setup do banco de dados (SQLAlchemy)."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)


def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False, future=True)


def init_db(engine: Engine | None = None) -> None:
    """Cria as tabelas se não existirem."""
    from src.db.models import Base

    Base.metadata.create_all(engine or get_engine())
