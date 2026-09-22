"""Testes da API (usa TestClient do FastAPI)."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "db_ok" in body


def test_predict_requires_text(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_batch_size_limit(client):
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422
