"""Configurações centralizadas do projeto."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


class Settings(BaseSettings):
    """Configuração via variáveis de ambiente / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/reviews"
    MODEL_PATH: str = str(MODELS_DIR / "best_model.pkl")
    API_URL: str = "http://localhost:8000"

    # Reprodutibilidade
    RANDOM_SEED: int = 42

    # Coleta
    DEFAULT_APP_ID: str = "com.whatsapp"
    DEFAULT_LANG: str = "pt"
    DEFAULT_COUNTRY: str = "br"


settings = Settings()

# Garante que os diretórios existam
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
