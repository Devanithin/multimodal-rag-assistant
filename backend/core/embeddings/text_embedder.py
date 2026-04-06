from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedder():
    embedder = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embedder