from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from core.retrieval.hybrid_search import hybrid_search
from config import settings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3
    )
    return llm


def ask_question(question: str) -> dict:
    llm = get_llm()

    # Use hybrid search instead of semantic only
    chunks = hybrid_search(question, k=8)

    if not chunks:
        return {
            "answer": "No relevant content found. Please upload some documents first.",
            "sources": []
        }

    # Build context from chunks
    context = "\n\n".join([chunk["content"] for chunk in chunks])

    # Extract sources
    sources = list(set([
        chunk["metadata"].get("filename", "unknown")
        for chunk in chunks
    ]))

    prompt_text = f"""You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the question. Be detailed and specific in your answer.

Context:
{context}

Question:
{question}

Answer:"""

    chain = llm | StrOutputParser()
    answer = chain.invoke(prompt_text)

    return {
        "answer": answer,
        "sources": sources
    }