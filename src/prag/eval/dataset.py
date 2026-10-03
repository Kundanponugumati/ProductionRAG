"""FinanceBench questions as evaluation items."""

from pathlib import Path

import polars as pl
from pydantic import BaseModel


class EvalQuestion(BaseModel):
    id: str
    question: str
    answer: str
    question_type: str  # metrics-generated | domain-relevant | novel-generated
    doc_name: str
    evidence_pages: list[int]  # 0-indexed, same as Chunk.page_num


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    rows = pl.read_parquet(path).sort("financebench_id").iter_rows(named=True)
    return [
        EvalQuestion(
            id=r["financebench_id"],
            question=r["question"],
            answer=r["answer"],
            question_type=r["question_type"],
            doc_name=r["doc_name"],
            evidence_pages=sorted({e["evidence_page_num"] for e in r["evidence"]}),
        )
        for r in rows
    ]
