"""Coleta reviews da Google Play Store e persiste em CSV + PostgreSQL.

Uso:
    python -m src.data.collect --app-id com.whatsapp --count 2000
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd
from google_play_scraper import Sort, reviews

from src.config import RAW_DIR, settings
from src.db.database import get_engine, init_db
from src.db.models import Review

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def collect_reviews(app_id: str, count: int, lang: str, country: str) -> pd.DataFrame:
    """Coleta ``count`` reviews de um app na Play Store."""
    logger.info("Coletando %s reviews do app '%s' (%s/%s)", count, app_id, lang, country)

    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count,
    )

    df = pd.DataFrame(result)
    if df.empty:
        raise RuntimeError("Nenhum review retornado — verifique o app_id / idioma / país.")

    df = df.rename(columns={
        "reviewId": "review_id",
        "userName": "user_name",
        "content": "text",
        "score": "rating",
        "at": "created_at",
        "reviewCreatedVersion": "app_version",
        "thumbsUpCount": "thumbs_up",
    })

    keep = ["review_id", "user_name", "text", "rating", "created_at", "app_version", "thumbs_up"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["app_id"] = app_id
    df["collected_at"] = datetime.utcnow()

    # Rotulação automática baseada no rating (fraca supervisão)
    df["sentiment"] = df["rating"].apply(_rating_to_sentiment)

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]
    return df


def _rating_to_sentiment(rating: int | None) -> str:
    """1-2 = NEGATIVO, 3 = NEUTRO, 4-5 = POSITIVO."""
    if rating is None:
        return "NEUTRO"
    if rating <= 2:
        return "NEGATIVO"
    if rating == 3:
        return "NEUTRO"
    return "POSITIVO"


def save_to_csv(df: pd.DataFrame, app_id: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"reviews_{app_id}_{ts}.csv"
    df.to_csv(path, index=False)
    logger.info("Salvo em %s (%s registros)", path, len(df))
    return str(path)


def save_to_db(df: pd.DataFrame) -> int:
    """Insere reviews no PostgreSQL, ignorando duplicatas pelo review_id."""
    engine = get_engine()
    init_db(engine)

    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.orm import Session

    inserted = 0
    with Session(engine) as session:
        for _, row in df.iterrows():
            stmt = insert(Review).values(
                review_id=row["review_id"],
                app_id=row["app_id"],
                user_name=row.get("user_name"),
                text=row["text"],
                rating=int(row["rating"]) if pd.notna(row["rating"]) else None,
                sentiment=row["sentiment"],
                app_version=row.get("app_version"),
                thumbs_up=int(row["thumbs_up"]) if pd.notna(row.get("thumbs_up")) else 0,
                created_at=row["created_at"],
                collected_at=row["collected_at"],
            ).on_conflict_do_nothing(index_elements=["review_id"])
            result = session.execute(stmt)
            inserted += result.rowcount or 0
        session.commit()

    logger.info("Inseridos %s reviews novos no banco", inserted)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta reviews da Play Store")
    parser.add_argument("--app-id", default=settings.DEFAULT_APP_ID)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--lang", default=settings.DEFAULT_LANG)
    parser.add_argument("--country", default=settings.DEFAULT_COUNTRY)
    parser.add_argument("--no-db", action="store_true", help="Não persistir no banco (só CSV)")
    args = parser.parse_args()

    df = collect_reviews(args.app_id, args.count, args.lang, args.country)
    save_to_csv(df, args.app_id)
    if not args.no_db:
        save_to_db(df)

    print("\n=== Distribuição de sentimentos ===")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
