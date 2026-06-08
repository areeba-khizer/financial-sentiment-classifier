"""API tests using a lightweight stubbed classifier.

These exercise request/response contracts without needing a trained model, so
they run fast in CI. End-to-end model quality is checked separately by training.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.predict as predict_module
from app.main import app


class _StubClassifier:
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, texts):
        return [
            {
                "label": "positive",
                "score": 0.97,
                "probabilities": {"negative": 0.01, "neutral": 0.02, "positive": 0.97},
            }
            for _ in texts
        ]


@pytest.fixture(autouse=True)
def stub_classifier(monkeypatch):
    stub = _StubClassifier()
    predict_module.get_classifier.cache_clear()
    monkeypatch.setattr("app.main.get_classifier", lambda: stub)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_labels(client):
    resp = client.get("/labels")
    assert resp.status_code == 200
    assert set(resp.json()["labels"]) == {"negative", "neutral", "positive"}


def test_predict_batch(client):
    payload = {"texts": ["Profits surged this quarter.", "Shares fell sharply."]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2
    first = body["predictions"][0]
    assert first["label"] == "positive"
    assert pytest.approx(sum(first["probabilities"].values()), rel=1e-3) == 1.0
    assert first["text"] == "Profits surged this quarter."


def test_predict_rejects_empty(client):
    resp = client.post("/predict", json={"texts": []})
    assert resp.status_code == 422
