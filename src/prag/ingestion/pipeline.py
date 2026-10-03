"""Ingestion orchestration: load → parse → clean → chunk → embed → store."""

import logging
import time
from dataclasses import dataclass

from prag.config import Settings
from prag.ingestion.chunker import TokenChunker
from prag.ingestion.cleaner import clean_text
from prag.ingestion.embedder import Embedder
from prag.ingestion.loader import load_documents
from prag.ingestion.parser import parse_pdf
from prag.ingestion.store import VectorStore
from prag.io import write_jsonl
from prag.models import Chunk, Page

log = logging.getLogger(__name__)

# Chunks embedded + upserted per round trip. Each batch is committed to Qdrant,
# so an interrupted run keeps everything stored before the failure.
UPSERT_BATCH = 256


@dataclass
class IngestStats:
    documents: int
    pages: int
    chunks: int
    points_in_collection: int
    seconds: float


def run_ingest(settings: Settings, limit: int | None = None) -> IngestStats:
    start = time.perf_counter()

    docs = load_documents(settings.pdf_dir, settings.questions_path)[:limit]
    log.info("Loaded %d documents", len(docs))

    pages: list[Page] = []
    for doc in docs:
        pages.extend(parse_pdf(doc)) # we are simply parsing the each page of the document. and them to pages list.
    write_jsonl(settings.processed_dir / "pages.jsonl", pages)
    log.info("Parsed %d pages", len(pages))

    chunker = TokenChunker(settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    chunks: list[Chunk] = []
    for page in pages:
        cleaned = page.model_copy(update={"text": clean_text(page.text)})
        chunks.extend(chunker.chunk_page(cleaned, docs_by_name[page.doc_name]))
    write_jsonl(settings.processed_dir / "chunks.jsonl", chunks)
    log.info("Created %d chunks", len(chunks))

    embedder = Embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.query_instruction,
    )
    store = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    store.ensure_collection(embedder.dim)

    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i : i + UPSERT_BATCH]
        store.upsert(batch, embedder.embed_documents([c.text for c in batch]))
        log.info("Embedded + stored %d/%d chunks", i + len(batch), len(chunks))

    return IngestStats(
        documents=len(docs),
        pages=len(pages),
        chunks=len(chunks),
        points_in_collection=store.count(),
        seconds=time.perf_counter() - start,
    )
