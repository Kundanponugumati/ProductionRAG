"""Find downloaded PDFs and attach their FinanceBench metadata."""

from pathlib import Path

import polars as pl

from prag.models import Document


def load_documents(pdf_dir: Path, questions_path: Path) -> list[Document]:
    meta = (
        pl.read_parquet(questions_path)
        .select("doc_name", "company", "doc_type", "doc_period")
        .unique("doc_name")
    )
    meta_by_name = {row["doc_name"]: row for row in meta.iter_rows(named=True)}

    docs = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        row = meta_by_name.get(path.stem)
        if row is None:
            raise ValueError(f"No metadata for {path.name} in {questions_path}")
        docs.append(Document(path=path, **row))
    return docs
