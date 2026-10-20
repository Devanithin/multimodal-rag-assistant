from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from config import settings
from functools import lru_cache
from langchain_openai import ChatOpenAI

# ----------for openrouoter model ----------
# @lru_cache(maxsize=1)
def get_llm():
    llm = ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=300,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Multimodal RAG Assistant",
        },
        # Asks OpenRouter to keep reasoning-model thinking out of the returned answer.
        extra_body={"reasoning": {"exclude": True}},
    )
    return llm


def _strip_reasoning(text: str) -> str:
    # Some reasoning models inline their thinking before a closing </think> tag (the opening tag is sometimes missing).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()

# @lru_cache(maxsize=1)
# def get_llm():
#     llm = ChatGoogleGenerativeAI(
#         model=settings.GEMINI_MODEL,
#         google_api_key=settings.GEMINI_API_KEY,
#         temperature=0.3
#     )
#     return llm


def generate_answer(question: str, chunks: list) -> dict:
    if not chunks:
        return {
            "answer": "No relevant content found. Please upload some documents first.",
            "sources": []
        }

    # Build context from chunks
    context = "\n\n---\n\n".join([chunk["content"] for chunk in chunks])

    # Extract sources
    sources = list(set([
        chunk["metadata"].get("filename", "unknown")
        for chunk in chunks
    ]))

    prompt_text = f"""You answer questions using only the document excerpts below.

Rules:
- Use only information stated in the excerpts. Do not use outside knowledge, even if you know the answer.
- If the excerpts do not contain the answer, say that the documents do not contain this information, and stop there.
- Do not add numbers, names, examples or claims that are not in the excerpts.
- Excerpts may include tables flattened into one value per line. Only report a table value if it is clear which row and column it belongs to.
- Answer the question directly. Give only the final answer, without showing your reasoning.

Document excerpts:
{context}

Question: {question}

Answer:"""

    chain = get_llm() | StrOutputParser()
    answer = _strip_reasoning(chain.invoke(prompt_text))

    return {
        "answer": answer,
        "sources": sources
    }