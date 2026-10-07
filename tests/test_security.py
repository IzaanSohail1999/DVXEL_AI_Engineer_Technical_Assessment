from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_missing_trusted_tenant_header_is_rejected():
    r = client.post("/v1/answer", json={"question": "refund policy", "org_override": "ember"})
    assert r.status_code == 400


def test_org_override_cannot_replace_trusted_header():
    r = client.post(
        "/v1/answer",
        headers={"X-Org-ID": "vertex"},
        json={"question": "refund policy", "org_override": "ember"},
    )
    assert r.status_code == 200
    assert r.json()["org_id"] == "vertex"
