from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.vector import vector_client

client = TestClient(app)


def test_fast_search_is_not_degraded():
    response = client.post(
        "/v1/answer",
        headers={"X-Org-ID": "vertex"},
        json={"question": "refund policy"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False
    assert body["sources"]


def test_transient_timeout_is_retried(monkeypatch):
    calls = {"n": 0}
    original = vector_client.search

    async def wrapped(query, tenant_id, top_k):
        calls["n"] += 1
        return await original(query, tenant_id, top_k)

    monkeypatch.setattr(vector_client, "search", wrapped)
    monkeypatch.setattr(vector_client, "fail_next", True)
    response = client.post(
        "/v1/answer",
        headers={"X-Org-ID": "vertex"},
        json={"question": "refund policy"},
    )
    assert response.status_code == 200
    body = response.json()
    assert calls["n"] == 2
    assert body["degraded"] is False
    assert body["sources"]
    assert all(item["org_id"] == "vertex" for item in body["sources"])


def test_slow_search_degrades_after_retries(monkeypatch):
    calls = {"n": 0}
    original = vector_client.search

    async def wrapped(query, tenant_id, top_k):
        calls["n"] += 1
        return await original(query, tenant_id, top_k)

    monkeypatch.setattr(vector_client, "search", wrapped)
    monkeypatch.setattr(vector_client, "delay_seconds", 1.0)
    response = client.post(
        "/v1/answer",
        headers={"X-Org-ID": "ember"},
        json={"question": "refund policy"},
    )
    assert response.status_code == 200
    body = response.json()
    assert calls["n"] == settings.max_retries + 1
    assert body["degraded"] is True
    assert body["sources"] == []
    assert body["org_id"] == "ember"
    assert "enough evidence" in body["answer"].lower()
