from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.llm.llm_client import generate_answer
from core.retrieval.hybrid_search import hybrid_search
import time

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    total_start = time.perf_counter()

    chunks = hybrid_search(request.question, k=8)
    retrieval_time = round(time.perf_counter() - total_start, 3)

    llm_start = time.perf_counter()
    result = generate_answer(request.question, chunks)
    llm_time = round(time.perf_counter() - llm_start, 3)

    total_time = round(time.perf_counter() - total_start, 3)

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