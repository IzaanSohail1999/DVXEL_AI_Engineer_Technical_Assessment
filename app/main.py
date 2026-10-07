import asyncio
from fastapi import FastAPI, Header, HTTPException
from .config import settings
from .schemas import AnswerRequest, AnswerResponse
from .vector import vector_client
from .llm import model_client

app = FastAPI(title="Dvxel AI Knowledge Service")


async def retrieve_with_retry(question: str, tenant_id: str, top_k: int):
    for _ in range(settings.max_retries + 1):
        try:
            return await asyncio.wait_for(
                vector_client.search(question, tenant_id, top_k),
                timeout=settings.vector_timeout_seconds,
            )
        except TimeoutError:
            continue
    return None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(
    payload: AnswerRequest,
    trusted_id: str | None = Header(default=None, alias="X-Org-ID"),
):
    if not trusted_id:
        raise HTTPException(status_code=400, detail="trusted tenant header is required")
    tenant_id = trusted_id
    evidence = await retrieve_with_retry(payload.question, tenant_id, payload.top_k)
    if evidence is None:
        answer_text = await model_client.answer(payload.question, [])
        return AnswerResponse(
            org_id=tenant_id,
            answer=answer_text,
            sources=[],
            degraded=True,
        )
    answer_text = await model_client.answer(payload.question, evidence)
    return AnswerResponse(
        org_id=tenant_id,
        answer=answer_text,
        sources=evidence,
        degraded=False,
    )
