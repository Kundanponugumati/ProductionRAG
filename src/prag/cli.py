"""Command line: `uv run prag ingest | search | stats | eval-retrieval`."""

import logging

import typer

from prag.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def ingest(
    limit: int | None = typer.Option(None, help="Only ingest the first N documents."),
) -> None:
    """Parse, chunk, embed and store all downloaded PDFs."""
    from prag.ingestion.pipeline import run_ingest  # heavy imports (torch) only when needed

    _setup_logging()
    stats = run_ingest(get_settings(), limit)
    typer.echo(
        f"\n{stats.documents} docs → {stats.pages} pages → {stats.chunks} chunks "
        f"in {stats.seconds / 60:.1f} min. Points in collection: {stats.points_in_collection}"
    )


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, "-k", help="Number of results."),
) -> None:
    """Embed a question and print the top-k chunks."""
    from prag.ingestion.embedder import Embedder
    from prag.ingestion.store import VectorStore

    s = get_settings()
    embedder = Embedder(
        s.embedding_model, s.embedding_device, query_instruction=s.query_instruction
    )
    store = VectorStore(s.qdrant_url, s.qdrant_collection)

    for rank, result in enumerate(store.search(embedder.embed_query(query), k), 1):
        c = result.chunk
        snippet = " ".join(c.text.split())[:200]
        typer.echo(f"\n#{rank}  score={result.score:.3f}  {c.doc_name}  page={c.page_num}")
        typer.echo(f"    {snippet}")


@app.command()
def stats() -> None:
    """Show point count in Qdrant vs chunks on disk."""
    from prag.ingestion.store import VectorStore

    s = get_settings()
    chunks_path = s.processed_dir / "chunks.jsonl"
    on_disk = sum(1 for _ in chunks_path.open()) if chunks_path.exists() else 0
    points = VectorStore(s.qdrant_url, s.qdrant_collection).count()
    typer.echo(f"collection '{s.qdrant_collection}': {points} points | chunks.jsonl: {on_disk}")


@app.command("eval-retrieval")
def eval_retrieval(
    name: str = typer.Option("run", help="Label for this run, e.g. 'baseline'."),
) -> None:
    """Run all FinanceBench questions through search and score retrieval."""
    from prag.eval.dataset import load_eval_questions
    from prag.eval.retrieval import evaluate_retrieval, save_run, summarize
    from prag.ingestion.embedder import Embedder
    from prag.ingestion.store import VectorStore

    s = get_settings()
    questions = load_eval_questions(s.questions_path)
    embedder = Embedder(
        s.embedding_model, s.embedding_device, query_instruction=s.query_instruction
    )
    store = VectorStore(s.qdrant_url, s.qdrant_collection)
    results = evaluate_retrieval(questions, embedder, store)

    summary = summarize(results)
    config = {
        "embedding_model": s.embedding_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "collection": s.qdrant_collection,
    }
    run_dir = save_run(results, summary, config, s.eval_dir, name)

    typer.echo(f"\nRetrieval eval — {summary['questions']} questions")
    for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "doc_hit@5"):
        typer.echo(f"  {key:<20} {summary[key]:.3f}")
    typer.echo(
        f"  {'answer_in_context@5':<20} {summary['answer_in_context@5']:.3f}"
        f"  (on {summary['answer_in_context_n']} single-number answers)"
    )

    typer.echo("\nBy question type:")
    for qtype in sorted({r.question_type for r in results}):
        sub = summarize([r for r in results if r.question_type == qtype])
        typer.echo(
            f"  {qtype:<18} n={sub['questions']:<3} hit@5={sub['hit@5']:.2f}  "
            f"mrr={sub['mrr']:.2f}  doc_hit@5={sub['doc_hit@5']:.2f}"
        )

    typer.echo(f"\nSaved → {run_dir}")
