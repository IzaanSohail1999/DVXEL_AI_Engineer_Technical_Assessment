import pytest
from app.vector import VectorStoreClient, vector_client


@pytest.mark.asyncio
async def test_vertex_search_excludes_ember_chunks():
    results = await vector_client.search("refund policy", "vertex", 5)
    assert results
    assert all(item.org_id == "vertex" for item in results)
    assert all(item.chunk_id.startswith("vx") for item in results)


@pytest.mark.asyncio
async def test_ember_search_excludes_vertex_chunks():
    results = await vector_client.search("refund policy", "ember", 5)
    assert results
    assert all(item.org_id == "ember" for item in results)
    assert all(item.chunk_id.startswith("em") for item in results)


@pytest.mark.asyncio
async def test_refund_question_ranks_refund_chunk_first():
    results = await vector_client.search("How long are refunds allowed after capture?", "vertex", 3)
    assert results[0].chunk_id == "vx1"


@pytest.mark.asyncio
async def test_top_k_is_applied_after_ranking():
    results = await vector_client.search("When are settlement exports produced?", "vertex", 1)
    assert [item.chunk_id for item in results] == ["vx2"]


@pytest.mark.asyncio
async def test_stored_score_breaks_equal_overlap():
    client = VectorStoreClient()
    client.delay_seconds = 0
    client.docs = {
        "vertex": [
            {
                "chunk_id": "low",
                "org_id": "vertex",
                "content": "refund policy",
                "metadata": {"keywords": ["refund"]},
                "score": 0.2,
            },
            {
                "chunk_id": "high",
                "org_id": "vertex",
                "content": "refund policy",
                "metadata": {"keywords": ["refund"]},
                "score": 0.9,
            },
        ]
    }
    results = await client.search("refund", "vertex", 2)
    assert [item.chunk_id for item in results] == ["high", "low"]
