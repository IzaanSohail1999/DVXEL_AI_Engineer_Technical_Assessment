import pytest
from app.evaluation import answer_is_grounded, retrieval_mrr, retrieval_recall_at_k
from app.llm import model_client
from app.schemas import Evidence


def _evidence(chunk_id="good", content="Refunds are allowed within 7 calendar days of settlement.", score=0.9):
    return Evidence(chunk_id=chunk_id, org_id="vertex", content=content, score=score)


def test_retrieval_metric_does_not_count_irrelevant_evidence_as_relevant():
    assert retrieval_recall_at_k([_evidence("wrong", "Unrelated content.")], ["good"]) == 0.0


def test_retrieval_recall_is_one_when_every_relevant_id_is_retrieved():
    retrieved = [_evidence("vx1"), _evidence("vx2"), _evidence("vx3")]
    assert retrieval_recall_at_k(retrieved, ["vx1", "vx2"]) == 1.0


def test_retrieval_recall_is_the_fraction_of_relevant_ids_found():
    retrieved = [_evidence("vx1"), _evidence("other")]
    assert retrieval_recall_at_k(retrieved, ["vx1", "vx2"]) == 0.5


def test_mrr_scores_the_rank_of_the_first_relevant_chunk():
    assert retrieval_mrr([_evidence("good"), _evidence("other")], ["good"]) == 1.0
    assert retrieval_mrr([_evidence("other"), _evidence("good")], ["good"]) == 0.5
    assert retrieval_mrr([_evidence("other"), _evidence("wrong")], ["good"]) == 0.0


def test_grounding_metric_rejects_unsupported_claim():
    source = "Refunds are allowed within 7 calendar days of settlement."
    answer = source + " Refunds are always automatic."
    assert answer_is_grounded(answer, [_evidence("good", source)], ["7 calendar days"]) is False


def test_grounding_accepts_answer_built_only_from_evidence():
    first = "Refunds are allowed within 7 calendar days of settlement."
    second = "Settlement reports are generated daily at 02:00 UTC."
    answer = f"{first} {second}"
    assert answer_is_grounded(
        answer,
        [_evidence("first", first), _evidence("second", second)],
        ["7 calendar days", "02:00 UTC"],
    ) is True


def test_grounding_rejects_answer_missing_a_required_fact():
    source = "Refunds are allowed within 7 calendar days of settlement."
    assert answer_is_grounded(source, [_evidence("good", source)], ["02:00 UTC"]) is False


@pytest.mark.asyncio
async def test_multi_fact_answer_uses_multiple_retrieved_evidence_items():
    first = _evidence("first", "Refunds are allowed within 7 calendar days of settlement.", 0.9)
    second = _evidence("second", "Settlement reports are generated daily at 02:00 UTC.", 0.8)
    answer = await model_client.answer("Give me the refund window and settlement report time.", [first, second])
    assert "7 calendar days" in answer
    assert "02:00 UTC" in answer
