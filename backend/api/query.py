from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.llm.llm_client import ask_question
import time

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ── Track total time ──
    total_start = time.time()

    # ── Track retrieval time ──
    retrieval_start = time.time()
    from core.retrieval.vector_store import search_chunks
    chunks = search_chunks(request.question)
    retrieval_time = round(time.time() - retrieval_start, 3)

    # ── Track LLM time ──
    llm_start = time.time()
    result = ask_question(request.question)
    llm_time = round(time.time() - llm_start, 3)

    total_time = round(time.time() - total_start, 3)

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "latency": {
            "retrieval_seconds": retrieval_time,
            "llm_seconds": llm_time,
            "total_seconds": total_time
        }
    }


@router.get("/test-chunks")
async def test_chunks(query: str = "what is this document about", k: int = 5):
    from core.retrieval.vector_store import search_chunks
    results = search_chunks(query, k=k)
    return {
        "count": len(results),
        "chunks": [
            {
                "content": r.page_content[:300],
                "metadata": r.metadata
            }
            for r in results
        ]
    }