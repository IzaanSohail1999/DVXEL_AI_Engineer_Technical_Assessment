import asyncio
import time

import httpx
import pytest

from app.main import app
from app.vector import vector_client


@pytest.mark.asyncio
async def test_overlapping_requests_do_not_share_sources():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        vertex, ember = await asyncio.gather(
            client.post(
                "/v1/answer",
                headers={"X-Org-ID": "vertex"},
                json={"question": "refund policy", "top_k": 3},
            ),
            client.post(
                "/v1/answer",
                headers={"X-Org-ID": "ember"},
                json={"question": "refund policy", "top_k": 3},
            ),
        )

    assert vertex.status_code == 200
    assert ember.status_code == 200
    vertex_sources = vertex.json()["sources"]
    ember_sources = ember.json()["sources"]
    assert vertex_sources
    assert ember_sources

    ember_ids = []
    for source in ember_sources:
        ember_ids.append(source["chunk_id"])
        assert source["org_id"] == "ember"

    for source in vertex_sources:
        assert source["org_id"] == "vertex"
        assert source["chunk_id"] not in ember_ids


@pytest.mark.asyncio
async def test_overlapping_requests_do_not_block_each_other():
    delay = 0.15
    original_delay = vector_client.delay_seconds
    vector_client.delay_seconds = delay
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started = time.perf_counter()
            vertex, ember = await asyncio.gather(
                client.post(
                    "/v1/answer",
                    headers={"X-Org-ID": "vertex"},
                    json={"question": "refund policy"},
                ),
                client.post(
                    "/v1/answer",
                    headers={"X-Org-ID": "ember"},
                    json={"question": "refund policy"},
                ),
            )
            elapsed = time.perf_counter() - started
    finally:
        vector_client.delay_seconds = original_delay

    assert vertex.status_code == 200
    assert ember.status_code == 200
    assert vertex.json()["degraded"] is False
    assert ember.json()["degraded"] is False
    one_wait = delay
    two_waits = delay + delay
    halfway = (one_wait + two_waits) / 2
    assert elapsed < halfway
