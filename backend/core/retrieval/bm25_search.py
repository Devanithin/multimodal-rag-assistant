from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from core.embeddings.text_embedder import get_embedder
from config import settings


def get_all_chunks():
    embedder = get_embedder()
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embedder,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )
    data = vector_store.get()
    documents = data["documents"]
    metadatas = data["metadatas"]
    return documents, metadatas


def bm25_search(query: str, k: int = 8):
    documents, metadatas = get_all_chunks()

    if not documents:
        return []

    # Tokenize all documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    tokenized_query = query.lower().split()

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)

    # Get scores for all documents
    scores = bm25.get_scores(tokenized_query)

    # Get top k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Return results as list of dicts
    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append({
                "content": documents[i],
                "metadata": metadatas[i],
                "score": scores[i]
            })

    return results