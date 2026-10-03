"""Retrieval evaluation: did search bring back the evidence page?

Metrics (per question, then averaged):
  Hit@k             evidence page is among the top-k chunks
  MRR               1 / rank of the first evidence-page chunk (0 if not in top-K)
  Doc Hit@5         the right *document* is among the top-5 (wrong company/year → miss)
  Answer-in-context the reference number appears in the top-5 chunk texts
                    (only for answers that are a single number; None otherwise)
"""

import re
from datetime import datetime
from pathlib import Path

import orjson
from pydantic import BaseModel

from prag.eval.dataset import EvalQuestion
from prag.ingestion.embedder import Embedder
from prag.ingestion.store import VectorStore
from prag.io import write_jsonl
from prag.models import SearchResult

HIT_KS = (1, 3, 5, 10)
CONTEXT_K = 5  # chunks we'd hand to the LLM

# "$1577.00", "-0.02", "1.9%", "93.86" — a bare number, nothing else.
_NUMERIC_ANSWER = re.compile(r"^\s*\$?\s*(-?[\d,]*\.?\d+)\s*%?\s*$")
_NUMBER_IN_TEXT = re.compile(r"\d[\d,]*\.?\d*")


class Retrieved(BaseModel):
    doc_name: str
    page_num: int
    score: float


class QuestionResult(BaseModel):
    id: str
    question_type: str
    question: str
    answer: str
    doc_name: str
    evidence_pages: list[int]
    first_hit_rank: int | None  # 1-based rank of first evidence-page chunk
    doc_hit_at_5: bool
    answer_in_context: bool | None  # None = answer isn't a single number
    retrieved: list[Retrieved]


def _numeric_answer(answer: str) -> float | None:
    match = _NUMERIC_ANSWER.match(answer)
    if not match:
        return None
    value = abs(float(match.group(1).replace(",", "")))
    return value if value != 0 else None  # "0" matches almost any page


def _numbers_in(text: str) -> set[float]:
    # Filings print negatives as "(1,577)", so compare absolute values.
    return {float(n.replace(",", "").rstrip(".")) for n in _NUMBER_IN_TEXT.findall(text)}


def score_question(q: EvalQuestion, results: list[SearchResult]) -> QuestionResult:
    evidence = {(q.doc_name, p) for p in q.evidence_pages}
    first_hit_rank = next(
        (
            rank
            for rank, r in enumerate(results, 1)
            if (r.chunk.doc_name, r.chunk.page_num) in evidence
        ),
        None,
    )

    top = results[:CONTEXT_K]
    target = _numeric_answer(q.answer)
    answer_in_context = None
    if target is not None:
        found = set().union(*(_numbers_in(r.chunk.text) for r in top))
        answer_in_context = any(abs(n - target) < 1e-6 for n in found)

    return QuestionResult(
        id=q.id,
        question_type=q.question_type,
        question=q.question,
        answer=q.answer,
        doc_name=q.doc_name,
        evidence_pages=q.evidence_pages,
        first_hit_rank=first_hit_rank,
        doc_hit_at_5=any(r.chunk.doc_name == q.doc_name for r in top),
        answer_in_context=answer_in_context,
        retrieved=[
            Retrieved(doc_name=r.chunk.doc_name, page_num=r.chunk.page_num, score=r.score)
            for r in results
        ],
    )


def evaluate_retrieval(
    questions: list[EvalQuestion], embedder: Embedder, store: VectorStore
) -> list[QuestionResult]:
    top_k = max(HIT_KS)
    vectors = embedder.embed_queries([q.question for q in questions])
    return [
        score_question(q, store.search(v, top_k)) for q, v in zip(questions, vectors, strict=True)
    ]


def summarize(results: list[QuestionResult]) -> dict[str, float | int]:
    n = len(results)
    ranks = [r.first_hit_rank for r in results]
    summary: dict[str, float | int] = {"questions": n}
    for k in HIT_KS:
        summary[f"hit@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    summary["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    summary["doc_hit@5"] = sum(r.doc_hit_at_5 for r in results) / n

    checkable = [r.answer_in_context for r in results if r.answer_in_context is not None]
    summary["answer_in_context@5"] = sum(checkable) / len(checkable) if checkable else 0.0
    summary["answer_in_context_n"] = len(checkable)
    return summary


def save_run(
    results: list[QuestionResult],
    summary: dict[str, float | int],
    config: dict[str, object],
    out_dir: Path,
    name: str,
) -> Path:
    """Write per-question results + summary, and append the summary to history.jsonl."""
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    run_dir = out_dir / "runs" / run_id
    write_jsonl(run_dir / "results.jsonl", results)

    record = {"run_id": run_id, "config": config, "summary": summary}
    (run_dir / "summary.json").write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
    with (out_dir / "history.jsonl").open("ab") as f:
        f.write(orjson.dumps(record) + b"\n")
    return run_dir
