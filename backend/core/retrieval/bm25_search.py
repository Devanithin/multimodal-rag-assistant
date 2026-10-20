import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from core.retrieval.vector_store import get_vector_store, get_store_version

_TOKEN_RE = re.compile(r"\w+")
_lock = threading.Lock()
_cache = {"version": None, "bm25": None, "documents": [], "metadatas": []}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _get_index():
    # Read the version before fetching so a write during the build forces a rebuild next time.
    version = get_store_version()
    with _lock:
        if _cache["version"] != version:
            data = get_vector_store().get(include=["documents", "metadatas"])
            documents, metadatas = data["documents"], data["metadatas"]
            _cache["bm25"] = BM25Okapi([tokenize(d) for d in documents]) if documents else None
            _cache["documents"] = documents
            _cache["metadatas"] = metadatas
            _cache["version"] = version
        return _cache["bm25"], _cache["documents"], _cache["metadatas"]


def bm25_search(query: str, k: int = 8):
    bm25, documents, metadatas = _get_index()
    tokenized_query = tokenize(query)
    if bm25 is None or not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]

    return [
        {"content": documents[i], "metadata": metadatas[i], "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]
