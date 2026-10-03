"""Download a FinanceBench subset: question metadata + the PDFs they refer to.

Usage:
    uv run python scripts/download_financebench.py --docs 25

Outputs:
    data/raw/pdfs/<doc_name>.pdf
    data/raw/questions.parquet   (only questions whose doc was downloaded)
"""

from pathlib import Path

import httpx
import polars as pl
import typer
from huggingface_hub import hf_hub_download

HF_REPO = "PatronusAI/financebench"
HF_FILE = "financebench_merged.jsonl"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"

RAW_DIR = Path("data/raw")
PDF_DIR = RAW_DIR / "pdfs"


def load_questions() -> pl.DataFrame:
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    return pl.read_ndjson(path)


def pick_docs(questions: pl.DataFrame, n_docs: int) -> list[str]:
    """Docs with the most questions first; doc_name breaks ties so the subset is deterministic."""
    counts = (
        questions.group_by("doc_name").len().sort(["len", "doc_name"], descending=[True, False])
    )
    return counts["doc_name"].head(n_docs).to_list()


def download_pdf(client: httpx.Client, doc_name: str) -> Path:
    dest = PDF_DIR / f"{doc_name}.pdf"
    if dest.exists():
        return dest

    # Write to a temp file and rename, so an interrupted download never leaves a half-written PDF.
    tmp = dest.with_suffix(".pdf.part")
    with client.stream("GET", PDF_URL.format(doc_name=doc_name)) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    tmp.rename(dest)
    return dest


def main(docs: int = typer.Option(25, help="Number of documents to download (max 84).")) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    
    doc_names = pick_docs(questions, docs)

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for i, doc_name in enumerate(doc_names, 1):
            path = download_pdf(client, doc_name)
            size_mb = path.stat().st_size / 1e6
            print(f"[{i:>2}/{len(doc_names)}] {doc_name} ({size_mb:.1f} MB)")

    subset = questions.filter(pl.col("doc_name").is_in(doc_names))
    subset.write_parquet(RAW_DIR / "questions.parquet")
    print(f"\n{len(doc_names)} docs, {subset.height} questions → {RAW_DIR / 'questions.parquet'}")


if __name__ == "__main__":
    typer.run(main)
