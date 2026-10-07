import re
from .benchmark import BENCHMARK
from .llm import model_client
from .vector import vector_client


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())

# This function is used to find out how many of the relevant chunks are retrieved
def retrieval_recall_at_k(retrieved, relevant_chunk_ids):
    relevant = set()

    for chunk_id in relevant_chunk_ids:
        if chunk_id is not None and chunk_id != "":
            relevant.add(chunk_id)

    if not relevant:
        return 0.0

    retrieved_ids = set()
    for item in retrieved:
        chunk_id = getattr(item, "chunk_id", None)
        if chunk_id is not None and chunk_id != "":
            retrieved_ids.add(chunk_id)
            
    return len(relevant & retrieved_ids) / len(relevant)


# This function is used to find out the mean reciprocal rank of the relevant chunks as in the order of the retrieved chunks compared to the relevant chunks.
def retrieval_mrr(retrieved, relevant_chunk_ids):
    relevant = {chunk_id for chunk_id in relevant_chunk_ids if chunk_id}
    for index, item in enumerate(retrieved, start=1):
        if getattr(item, "chunk_id", None) in relevant:
            return 1.0 / index
    return 0.0

# This function is used to find out if the answer is grounded in the evidence, which means that all answers are supported by the evidence and also claims are supported by the evidence.
def answer_is_grounded(answer, evidence, required_facts):
    if not answer or not evidence:
        return False
    answer_norm = _normalize(answer)
    facts_present = all(_normalize(fact) in answer_norm for fact in required_facts)
    note_texts = []
    for item in evidence:
        content = getattr(item, "content", "")
        note_texts.append(content)
    evidence_text = " ".join(note_texts)
    evidence_norm = _normalize(evidence_text)
    if not evidence_norm:
        return False
    sentences = []
    for part in answer_norm.split("."):
        cleaned = part.strip()
        if cleaned != "":
            sentences.append(cleaned)
    claims_supported = True
    for sentence in sentences:
        if sentence not in evidence_norm:
            claims_supported = False
    return facts_present and claims_supported


async def evaluate_benchmark(top_k: int = 3):
    rows = []
    for case in BENCHMARK:
        retrieved = await vector_client.search(case["question"], case["tenant"], top_k)
        answer = await model_client.answer(case["question"], retrieved)
        rows.append({
            "id": case["id"],
            "retrieval_recall_at_k": retrieval_recall_at_k(retrieved, case["relevant_chunk_ids"]),
            "retrieval_mrr": retrieval_mrr(retrieved, case["relevant_chunk_ids"]),
            "grounded": answer_is_grounded(answer, retrieved, case["required_facts"]),
        })
    return {
        "n_cases": len(rows),
        "retrieval_recall_at_k": sum(r["retrieval_recall_at_k"] for r in rows) / len(rows),
        "retrieval_mrr": sum(r["retrieval_mrr"] for r in rows) / len(rows),
        "grounded_answer_rate": sum(r["grounded"] for r in rows) / len(rows),
        "cases": rows,
    }
