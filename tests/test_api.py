"""
API contract tests using FastAPI's TestClient — these run without an actual server
process (no port binding needed), which is exactly what you want in a CI pipeline
(see .github/workflows/ci.yml, which runs this file on every push).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_returns_valid_schema(client):
    response = client.post("/predict", json={"text": "This was such a helpful video, thanks!"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] in {"positive", "neutral", "negative"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0


def test_predict_endpoint_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_ask_endpoint_returns_valid_schema(client):
    response = client.post("/ask", json={"question": "What do people ask about?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert isinstance(body["sources"], list)


def test_ask_endpoint_rejects_invalid_sentiment_filter(client):
    response = client.post("/ask", json={"question": "test", "sentiment_filter": "furious"})
    assert response.status_code == 400


def test_drift_endpoint_responds(client):
    response = client.get("/monitor/drift")
    assert response.status_code == 200
    assert "status" in response.json()
