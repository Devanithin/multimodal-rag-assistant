from core.retrieval.vector_store import search_chunks
from core.retrieval.bm25_search import bm25_search

RRF_K = 60


def _chunk_key(content: str, metadata: dict):
    if "file_id" in metadata and "chunk_index" in metadata:
        return (metadata["file_id"], metadata["chunk_index"])
    return content


def hybrid_search(query: str, k: int = 8, fetch_k: int = 20) -> list:
    semantic_results = [
        (doc.page_content, doc.metadata) for doc in search_chunks(query, k=fetch_k)
    ]
    bm25_results = [
        (r["content"], r["metadata"]) for r in bm25_search(query, k=fetch_k)
    ]

    # Reciprocal Rank Fusion: rank-based, so the two retrievers' incomparable scores don't matter.
    fused = {}
    for retriever, results in (("semantic", semantic_results), ("bm25", bm25_results)):
        for rank, (content, metadata) in enumerate(results):
            key = _chunk_key(content, metadata)
            entry = fused.setdefault(key, {
                "content": content,
                "metadata": metadata,
                "score": 0.0,
                "retrievers": [],
            })
            entry["score"] += 1.0 / (RRF_K + rank + 1)
            entry["retrievers"].append(retriever)

    ranked = sorted(fused.values(), key=lambda c: c["score"], reverse=True)[:k]
    for chunk in ranked:
        retrievers = chunk.pop("retrievers")
        chunk["source"] = "both" if len(retrievers) > 1 else retrievers[0]
    return ranked
