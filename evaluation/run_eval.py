"""Evaluate retrieval and answer quality of the RAG pipeline against golden_dataset.json.

Run from anywhere:  python evaluation/run_eval.py [--fresh] [--ids att-01,neg-01]
"""
import argparse
import inspect
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import unicodedata
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent / "backend"
DOCS_DIR = EVAL_DIR / "docs"
DATASET_FILE = EVAL_DIR / "golden_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
PROGRESS_FILE = RESULTS_DIR / ".progress.json"
K = 8

# Must be set before importing backend config: the eval index lives in a throwaway
# Chroma dir so the app's real knowledge base is never read or modified.
CHROMA_DIR = tempfile.mkdtemp(prefix="rag_eval_chroma_")
os.environ["CHROMA_PERSIST_DIR"] = CHROMA_DIR
os.environ.setdefault("USER_AGENT", "rag-eval")
sys.path.insert(0, str(BACKEND_DIR))
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI  # noqa: E402

from config import settings  # noqa: E402
from api.files import PARSERS  # noqa: E402
from core.ingestion.chunker import chunk_text  # noqa: E402
from core.retrieval.vector_store import store_chunks, search_chunks  # noqa: E402
from core.retrieval.bm25_search import bm25_search  # noqa: E402
from core.retrieval.hybrid_search import hybrid_search  # noqa: E402
from core.llm.llm_client import generate_answer  # noqa: E402

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL") or settings.OPENROUTER_MODEL
CORRECTNESS_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

RETRIEVERS = {
    "semantic": lambda q: [d.page_content for d in search_chunks(q, k=K)],
    "bm25": lambda q: [r["content"] for r in bm25_search(q, k=K)],
    "hybrid": lambda q: [c["content"] for c in hybrid_search(q, k=K)],
}

JUDGE_PROMPT = """You are grading the answer of a document question-answering system.

Question:
{question}

Reference answer:
{reference}

Context the system retrieved from the documents:
<context>
{context}
</context>

System answer:
<answer>
{answer}
</answer>

Grade two things:
1. "correctness" compares the system answer with the reference answer.
   - "correct": contains the key facts of the reference and does not contradict it.
   - "partial": some key facts are missing or slightly wrong.
   - "incorrect": wrong, missing the point, or contradicts the reference.
   If the reference says the question cannot be answered from the documents, "correct" means the
   system said the information is not available, and "incorrect" means it gave a substantive answer.
2. "faithful" is true if every factual claim in the system answer is supported by the context above
   (saying that information is unavailable counts as supported), and false if it states anything the
   context does not support, even if that claim is true in the real world.

Respond with only a JSON object, no other text:
{{"correctness": "correct" | "partial" | "incorrect", "faithful": true | false, "reason": "<one sentence>"}}"""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).lower().strip()


def first_relevant_rank(contents: list[str], evidence: list[str]):
    phrases = [normalize(e) for e in evidence]
    for rank, content in enumerate(contents, start=1):
        text = normalize(content)
        if any(p in text for p in phrases):
            return rank
    return None


def build_index() -> dict[str, list[str]]:
    chunks_by_file = {}
    for file_id, path in enumerate(sorted(DOCS_DIR.iterdir()), start=1):
        ext = path.suffix.lower().lstrip(".")
        if ext not in PARSERS:
            continue
        chunks = chunk_text(PARSERS[ext](str(path)))
        store_chunks(chunks, file_id=file_id, filename=path.name)
        chunks_by_file[path.name] = chunks
        print(f"  indexed {path.name}: {len(chunks)} chunks")
    return chunks_by_file


def validate_dataset(questions, chunks_by_file):
    for q in questions:
        if q["type"] == "unanswerable":
            continue
        chunks = [normalize(c) for c in chunks_by_file.get(q["source"], [])]
        if not chunks:
            print(f"  WARNING {q['id']}: source '{q['source']}' is not in evaluation/docs")
        elif not any(normalize(e) in c for e in q["evidence"] for c in chunks):
            print(f"  WARNING {q['id']}: no evidence phrase appears whole in any chunk; it can never count as retrieved")


def evaluate_retrieval(questions):
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    summary = {}
    ranks_by_question = {q["id"]: {} for q in answerable}
    for name, retrieve in RETRIEVERS.items():
        ranks, times = [], []
        for q in answerable:
            start = time.perf_counter()
            contents = retrieve(q["question"])
            times.append(time.perf_counter() - start)
            rank = first_relevant_rank(contents, q["evidence"])
            ranks.append(rank)
            ranks_by_question[q["id"]][name] = rank
        summary[name] = {
            "hit@1": statistics.mean(r is not None and r <= 1 for r in ranks),
            "hit@3": statistics.mean(r is not None and r <= 3 for r in ranks),
            f"hit@{K}": statistics.mean(r is not None for r in ranks),
            "mrr": statistics.mean(1 / r if r else 0 for r in ranks),
            "median_ms": statistics.median(times) * 1000,
        }
    return summary, ranks_by_question


def get_judge():
    return ChatOpenAI(
        model=JUDGE_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )


def parse_verdict(raw: str):
    # Take the last flat JSON object so any reasoning text the model emits first is ignored.
    for candidate in reversed(re.findall(r"\{[^{}]*\}", raw)):
        try:
            verdict = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if verdict.get("correctness") in CORRECTNESS_SCORES and isinstance(verdict.get("faithful"), bool):
            return verdict
    return {"correctness": None, "faithful": None, "reason": None, "judge_error": raw[:500]}


def judge(llm, q, chunks, answer):
    if q["type"] == "unanswerable":
        reference = "NONE. This question cannot be answered from the documents; the correct behavior is to say the information is not available."
    else:
        reference = q["reference_answer"]
    context = "\n\n---\n\n".join(c["content"] for c in chunks) or "(nothing retrieved)"
    prompt = JUDGE_PROMPT.format(question=q["question"], reference=reference, context=context, answer=answer)
    return parse_verdict(llm.invoke(prompt).content)


def run_config(use_judge: bool):
    params = inspect.signature(chunk_text).parameters
    return {
        "llm_model": settings.OPENROUTER_MODEL,
        "judge_model": JUDGE_MODEL if use_judge else None,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": params["chunk_size"].default,
        "chunk_overlap": params["chunk_overlap"].default,
        "k": K,
    }


def load_progress(config, fresh: bool) -> dict:
    if fresh or not PROGRESS_FILE.exists():
        return {}
    saved = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    if saved.get("config") != config:
        print("  Saved progress is from a different configuration; starting over.")
        return {}
    print(f"  Resuming: {len(saved['results'])} answers already evaluated.")
    return saved["results"]


def save_progress(config, results):
    PROGRESS_FILE.write_text(json.dumps({"config": config, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_generation(questions, config, results, use_judge: bool):
    llm = get_judge() if use_judge else None
    for i, q in enumerate(questions, start=1):
        done = results.get(q["id"])
        if done and done["question"] == q["question"]:
            continue
        chunks = hybrid_search(q["question"], k=K)
        start = time.perf_counter()
        output = generate_answer(q["question"], chunks)
        answer_seconds = time.perf_counter() - start
        if use_judge:
            verdict = judge(llm, q, chunks, output["answer"])
        else:
            verdict = {"correctness": None, "faithful": None, "reason": None}
        results[q["id"]] = {
            "question": q["question"],
            "answer": output["answer"],
            "sources": output["sources"],
            "retrieved": [(c["metadata"].get("filename"), c["metadata"].get("chunk_index")) for c in chunks],
            "context": [c["content"] for c in chunks],
            "answer_seconds": round(answer_seconds, 2),
            **verdict,
        }
        save_progress(config, results)
        if use_judge:
            status = f"{verdict['correctness'] or 'JUDGE PARSE ERROR'}, faithful={verdict['faithful']}"
        else:
            status = f"answered in {answer_seconds:.1f}s"
        print(f"  [{i}/{len(questions)}] {q['id']}: {status}")


def apply_grades(results_path: Path, grades_path: Path):
    """Merge grades from an external judge (a person or another model) into a results file."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    grades = json.loads(grades_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in data["questions"]}
    missing = set(by_id) - set(grades["grades"])
    if missing:
        sys.exit(f"Grades file is missing: {', '.join(sorted(missing))}")
    for qid, grade in grades["grades"].items():
        if grade["correctness"] not in CORRECTNESS_SCORES or not isinstance(grade["faithful"], bool):
            sys.exit(f"Invalid grade for {qid}: {grade}")
        by_id[qid].update(correctness=grade["correctness"], faithful=grade["faithful"], reason=grade.get("reason"))

    data["config"]["judge_model"] = grades["judge"]
    data["generation"] = summarize_generation(data["questions"], by_id)
    results_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    ranks = {q["id"]: q["retrieval_ranks"] for q in data["questions"] if q["retrieval_ranks"]}
    print_report(data["questions"], data["retrieval"], ranks, data["generation"], by_id, grades["judge"])
    print(f"\nUpdated {results_path}")


def summarize_generation(questions, results):
    answerable = [results[q["id"]] for q in questions if q["type"] != "unanswerable"]
    unanswerable = [results[q["id"]] for q in questions if q["type"] == "unanswerable"]
    graded = [r for r in answerable if r["correctness"] is not None]
    graded_neg = [r for r in unanswerable if r["correctness"] is not None]
    faithful = [r["faithful"] for r in answerable + unanswerable if r["faithful"] is not None]
    return {
        "answer_correctness": statistics.mean(CORRECTNESS_SCORES[r["correctness"]] for r in graded) if graded else None,
        "correctness_counts": dict(Counter(r["correctness"] for r in graded)),
        "abstention_rate": statistics.mean(r["correctness"] == "correct" for r in graded_neg) if graded_neg else None,
        "faithfulness": statistics.mean(faithful) if faithful else None,
        "judge_errors": sum(r["correctness"] is None for r in answerable + unanswerable),
        "median_answer_seconds": statistics.median(r["answer_seconds"] for r in answerable + unanswerable),
    }


def pct(value):
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def print_report(questions, retrieval, ranks, generation, results, judge_label):
    n_ans = sum(q["type"] != "unanswerable" for q in questions)
    n_neg = len(questions) - n_ans
    print(f"\nRETRIEVAL ({n_ans} answerable questions, top-{K})")
    print(f"  {'retriever':<10} {'hit@1':>7} {'hit@3':>7} {f'hit@{K}':>7} {'MRR':>6} {'median':>9}")
    for name, m in retrieval.items():
        print(f"  {name:<10} {pct(m['hit@1']):>7} {pct(m['hit@3']):>7} {pct(m[f'hit@{K}']):>7} {m['mrr']:6.3f} {m['median_ms']:7.1f}ms")

    print(f"\nANSWERS ({n_ans} answerable, {n_neg} unanswerable; judge: {judge_label or 'none yet, grade with --apply-grades'})")
    print(f"  correctness (answerable):    {pct(generation['answer_correctness'])}  {generation['correctness_counts']}")
    print(f"  abstention (unanswerable):   {pct(generation['abstention_rate'])}")
    print(f"  faithfulness (all):          {pct(generation['faithfulness'])}")
    print(f"  median answer time:          {generation['median_answer_seconds']:.1f}s")
    if generation["judge_errors"]:
        print(f"  judge parse errors:          {generation['judge_errors']}")

    failures = []
    for q in questions:
        r = results[q["id"]]
        problems = []
        if q["type"] != "unanswerable" and ranks[q["id"]]["hybrid"] is None:
            problems.append("evidence not retrieved")
        if r["correctness"] not in (None, "correct"):
            problems.append("answered instead of declining" if q["type"] == "unanswerable" else r["correctness"])
        if r["faithful"] is False:
            problems.append("unfaithful")
        if problems:
            failures.append(f"  {q['id']:<7} {', '.join(problems):<40} {q['question']}")
    if failures:
        print(f"\nFAILURES ({len(failures)})")
        print("\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true", help="discard saved progress and re-run every answer")
    parser.add_argument("--ids", help="comma-separated question ids to run (partial run; no results file is written)")
    parser.add_argument("--no-judge", action="store_true", help="only generate answers; grade them later with --apply-grades")
    parser.add_argument("--apply-grades", nargs=2, metavar=("RESULTS_JSON", "GRADES_JSON"),
                        help="merge grades from an external judge into a results file and print the report")
    args = parser.parse_args()

    try:
        if args.apply_grades:
            apply_grades(Path(args.apply_grades[0]), Path(args.apply_grades[1]))
            return

        if not settings.OPENROUTER_API_KEY:
            sys.exit("OPENROUTER_API_KEY is not set (backend/.env).")

        questions = json.loads(DATASET_FILE.read_text(encoding="utf-8"))["questions"]
        if args.ids:
            wanted = set(args.ids.split(","))
            questions = [q for q in questions if q["id"] in wanted]
        RESULTS_DIR.mkdir(exist_ok=True)
        use_judge = not args.no_judge
        config = run_config(use_judge)

        print("Building evaluation index...")
        chunks_by_file = build_index()
        validate_dataset(questions, chunks_by_file)

        print("Evaluating retrieval...")
        retrieval, ranks = evaluate_retrieval(questions)

        calls = "2 LLM calls each" if use_judge else "answers only, no judge"
        print(f"Evaluating answers ({len(questions)} questions, {calls})...")
        results = load_progress(config, args.fresh)
        try:
            evaluate_generation(questions, config, results, use_judge)
        except Exception as e:
            sys.exit(f"\nStopped by an LLM error: {type(e).__name__}: {e}\n"
                     f"Progress is saved; run the same command again to resume.")

        generation = summarize_generation(questions, results)
        print_report(questions, retrieval, ranks, generation, results, config["judge_model"])

        if args.ids:
            print("\nPartial run (--ids): no results file written.")
            return
        out = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": config,
            "retrieval": retrieval,
            "generation": generation,
            "questions": [
                {"id": q["id"], "type": q["type"], "reference_answer": q["reference_answer"],
                 "retrieval_ranks": ranks.get(q["id"]), **results[q["id"]]}
                for q in questions
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        PROGRESS_FILE.unlink(missing_ok=True)
        print(f"\nSaved {out.relative_to(EVAL_DIR.parent)}")
    finally:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
