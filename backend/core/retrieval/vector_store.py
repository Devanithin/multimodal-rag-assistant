from langchain_community.vectorstores import Chroma
from core.embeddings.text_embedder import get_embedder
from config import settings
from functools import lru_cache

# Bumped on every write so in-memory indexes built from the store (BM25) know to rebuild.
_store_version = 0


def get_store_version() -> int:
    return _store_version


def _bump_version():
    global _store_version
    _store_version += 1


@lru_cache(maxsize=1)
def get_vector_store():
    embedder = get_embedder()
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embedder,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )
    return vector_store


def store_chunks(chunks: list[str], file_id: int, filename: str):
    vector_store = get_vector_store()
    metadatas = [
        {"file_id": file_id, "filename": filename, "chunk_index": i}
        for i, _ in enumerate(chunks)
    ]
    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    _bump_version()


def search_chunks(query: str, k: int = 8):
    vector_store = get_vector_store()
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception:
        results = vector_store.similarity_search(query, k=3)
        return results


def delete_file_chunks(file_id: int):
    vector_store = get_vector_store()
    vector_store.delete(where={"file_id": file_id})
    _bump_version()