from core.retrieval.vector_store import search_chunks
from core.retrieval.bm25_search import bm25_search


def hybrid_search(query: str, k: int = 8) -> list:
    # ── Semantic Search ──
    semantic_results = search_chunks(query, k=k)
    semantic_contents = set()
    final_chunks = []

    for doc in semantic_results:
        semantic_contents.add(doc.page_content)
        final_chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": "semantic"
        })

    # ── BM25 Search ──
    bm25_results = bm25_search(query, k=k)

    for result in bm25_results:
        # Only add if not already in semantic results
        if result["content"] not in semantic_contents:
            final_chunks.append({
                "content": result["content"],
                "metadata": result["metadata"],
                "source": "bm25"
            })

    return final_chunks[:k]