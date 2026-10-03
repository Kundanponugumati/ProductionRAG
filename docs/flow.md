# Pipeline Flow

```
HuggingFace + GitHub
        │  1. download
        ▼
data/raw/pdfs/*.pdf + questions.parquet
        │  2. load
        ▼
list[Document]
        │  3. parse
        ▼
data/processed/pages.jsonl
        │  4. clean
        ▼
list[Page] (clean text)
        │  5. chunk
        ▼
data/processed/chunks.jsonl
        │  6. embed
        ▼
vectors
        │  7. store
        ▼
Qdrant collection "financebench"
        │  8. search
        ▼
top-k chunks for a query
```

## Steps

| # | Step | Code | Input → Output | Status |
|---|---|---|---|---|
| 1 | Download | `scripts/download_financebench.py` | FinanceBench (HF + GitHub) → 25 PDFs + `questions.parquet` | ✅ |
| 2 | Load | `ingestion/loader.py` → `load_documents()` | PDFs + parquet → `Document` (path + company, type, year) | ✅ |
| 3 | Parse | `ingestion/parser.py` → `parse_pdf()` | `Document` → one `Page` per PDF page (0-indexed) | ✅ |
| 4 | Clean | `ingestion/cleaner.py` → `clean_text()` | raw page text → normalised text | ✅ |
| 5 | Chunk | `ingestion/chunker.py` → `TokenChunker.chunk_page()` | `Page` → `Chunk`s of ≤500 BGE tokens, 64 overlap, within the page | ✅ |
| 6 | Embed | `ingestion/embedder.py` → `Embedder` | `Chunk` text → 1024-dim vectors (local bge-large, MPS) | ✅ |
| 7 | Store | `ingestion/store.py` → `VectorStore` | `Chunk` + vector → Qdrant point (UUIDv5 id, chunk as payload) | ✅ |
| 8 | Run / Search | `ingestion/pipeline.py` → `run_ingest()`, `cli.py` | `prag ingest` runs 2–7; `prag search` embeds a query → top-k chunks; `prag stats` | ✅ |
| 9 | Evaluate retrieval | `eval/dataset.py`, `eval/retrieval.py` | 84 questions → top-10 → Hit@k, MRR, Doc Hit@5 → `data/eval/` | ✅ |

## Shared pieces

| Code | Role |
|---|---|
| `config.py` | Settings from `.env` (model, chunk size, Qdrant URL, paths) |
| `models.py` | `Document` → `Page` → `Chunk` data shapes |
| `io.py` | Read/write the JSONL files between steps |

## Commands

```bash
uv run python scripts/download_financebench.py --docs 25   # step 1
uv run prag ingest [--limit N]                              # steps 2–7
uv run prag search "your question" -k 5                     # step 8
uv run prag stats                                           # points vs chunks
uv run prag eval-retrieval --name baseline                  # score retrieval on 84 questions
```
