# Iteration 1 — Ingestion MVP

> Goal: turn FinanceBench PDFs into searchable vectors in Qdrant with the **simplest pipeline that works**.
> No optimisation yet — every layer is deliberately naive so later iterations can improve it *with measurements*.

## Where this sits in the roadmap

```
INGESTION (offline)                                  RETRIEVAL + GENERATION (online)
PDFs → parse → clean → chunk → embed → [Qdrant]  ←  embed query ← user question
                                          ↓
                                   top-k chunks → prompt → LLM → answer
```

| Iter | Theme |
|---|---|
| **1** | **Ingestion MVP** (this doc) |
| 2 | Retrieval + generation MVP |
| 3 | Eval harness — baseline numbers (Hit@k, MRR, LLM-judge correctness) |
| 4 | Parsing quality (tables, layout, metadata) |
| 5 | Chunking strategies |
| 6 | Hybrid search (dense + sparse) + reranking |
| 7 | Query understanding (rewrite, decompose, filters) |
| 8 | Ingestion at scale (incremental, async, queues, versioning) |
| 9 | Serving (FastAPI, streaming, caching) |
| 10 | Observability + CI eval gates |

## Stack

| Concern | Tool | Why |
|---|---|---|
| Env / packages | uv (Rust) | fast installs, lockfile, `uv run` |
| Lint / format | ruff (Rust) | one tool for everything |
| Models / config | pydantic v2 (Rust core), pydantic-settings | typed data, `.env` config |
| JSON | orjson (Rust) | fast JSONL artifacts |
| Dataframes | polars (Rust) | dataset metadata, later eval results |
| PDF parsing | PyMuPDF (C) | fast page-level text extraction |
| Tokenizer | HF `tokenizers` (Rust) | the embedding model's own tokenizer → exact token budget |
| Embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers (local, MPS) | free, strong open-source baseline, 1024-dim |
| Vector DB | Qdrant (Rust) in Docker | hybrid search later without migration |
| Retries | tenacity | rate limits / transient errors |
| CLI | typer | `uv run prag ...` |
| Tests | pytest | |

## Dataset — FinanceBench

- Questions + metadata: Hugging Face `PatronusAI/financebench` (~150 Qs, each with `doc_name`, answer, evidence text and **evidence page number**).
- PDFs: GitHub `patronus-ai/financebench` → `pdfs/`.
- MVP subset: top ~25 docs by question count.
- **`evidence_page_num` is 0-indexed** and matches PyMuPDF's `doc[i]` (verified: 99/104 evidence snippets found on exactly that page, none on page+1). So we store `page_num` 0-indexed too.
- Subset with `--docs 25`: 25 PDFs (~40 MB), **84 of 150 questions**.
- We don't *train* anything — the questions are our **evaluation set** (used from iteration 3). The evidence page numbers give us free retrieval labels.

## Design decisions (and their known weaknesses)

| Decision | Why for MVP | Weakness to fix later |
|---|---|---|
| Plain page text via PyMuPDF | fast, zero config | tables become word soup (iter 4) |
| Chunk **within a page** only | each chunk maps to one page → trivial eval vs evidence page | sentences split across pages are cut (iter 5) |
| Fixed token window (500 BGE tokens, 64 overlap) | simple; fits BGE's 512 limit incl. special tokens | ignores section structure (iter 5) |
| Chunk text sliced from page via token offsets | BGE tokenizer lowercases/splits numbers, so decoding would corrupt text | — |
| Dense-only vectors | one moving part | weak on exact terms like tickers/line items (iter 6) |
| Deterministic UUIDv5 point IDs from `doc:page:chunk` | re-runs overwrite instead of duplicating → idempotent | re-embeds everything every run (iter 8: content hashing) |
| Write `pages.jsonl` / `chunks.jsonl` | every stage is inspectable | — |

## Project layout

```
ProductionRAG/
  pyproject.toml  uv.lock  .env.example  .gitignore  docker-compose.yml
  docs/iter-1.md
  scripts/download_financebench.py
  src/prag/
    config.py            # settings from .env
    models.py            # Document, Page, Chunk
    io.py                # JSONL read/write (orjson)
    ingestion/
      loader.py          # PDFs + FinanceBench metadata → list[Document]
      parser.py          # PDF → list[Page]
      cleaner.py         # minimal text normalisation
      chunker.py         # token window per page
      embedder.py        # batched OpenAI embeddings + retries
      store.py           # Qdrant collection / upsert / search
      pipeline.py        # orchestrates, writes data/processed/*.jsonl
    cli.py               # ingest | search | stats
  data/raw/  data/processed/   (gitignored)
  tests/
```

## Steps

Each step is small and reviewed before moving on.

- [x] **1. Scaffold** — `git init`, `uv init --package`, deps, ruff config, `.env.example`, `docker-compose.yml` (Qdrant).
- [x] **2. Dataset script** — download FinanceBench metadata, choose top-N docs, download PDFs, save `questions.parquet`.
- [x] **3. Models + config** — `Document`, `Page`, `Chunk`, `Settings`, plus `io.py` (JSONL) and `loader.py` (PDF + metadata).
- [x] **4. Parser** — PyMuPDF page text → `pages.jsonl`. *Result: 25 docs → 3,563 pages in ~10.6s, 5 empty pages. Tables come out as one token per line — expected, fixed in iter 4.*
- [x] **5. Cleaner + chunker** — normalise text, BGE-tokenizer sliding window per page → `chunks.jsonl`. *Result: 6,869 chunks, 2.65M tokens, chunking in 1.7s; every chunk is a verbatim substring of its page. 13 chunks <20 tokens (page-number-only pages) — kept for now.*
- [x] **6. Embedder** — local `bge-large-en-v1.5` on MPS, normalised vectors, BGE query instruction on queries only. *Result: ~10.6 chunks/s on M5 → ≈11 min for all 6,869 chunks, $0. Model load ~40s first time (download).*
- [x] **7. Store** — ensure collection (cosine, dim from model, guards against dim mismatch), upsert with UUIDv5 IDs + full payload, `search`, `count`.
  *Smoke test on 3M_2018_10K (337 chunks, temp collection): re-upsert keeps count at 337 ✅. Capex query → rank 1 is p38, which **does contain $1,577M**; the labelled evidence page 59 (cash-flow table) is only **rank 34** — broken table text embeds poorly.*
  *Takeaways for iter 3/4: (a) evidence-page Hit@k is strict — also check whether the answer appears in retrieved text; (b) table parsing is a real retrieval problem, not cosmetic.*
- [x] **8. Pipeline + CLI** — `prag ingest [--limit N]`, `prag search "q" -k 5`, `prag stats`.
  *Full run: 25 docs → 3,563 pages → 6,869 chunks → 6,869 points in 10.8 min (embedding ≈ 99% of the time). 337 points pre-loaded for the dashboard were overwritten, not duplicated ✅.*
  *Sample searches: 3M FY2018 capex → right doc, answer on p38 ✅. "3M 2022 segment" → **wrong year** (3M_2018_10K). "JnJ discontinued segment" → **wrong company** (all Pfizer). Pure semantic search ignores company/year → metadata filtering / query understanding (iter 6–7).*
  *Known limitation: `--limit N` rewrites `pages.jsonl` / `chunks.jsonl` with only N docs (Qdrant keeps the rest), so `prag stats` will show a mismatch afterwards.*
- [ ] **9. Tests** — chunker (overlap, token limits, page mapping), parser on a tiny PDF.

**Out of scope for iter 1:** answer generation, hybrid search, reranking, table parsing, async, API, eval metrics.

## Done when

1. `docker compose up -d` → Qdrant dashboard at http://localhost:6333/dashboard
2. `uv run python scripts/download_financebench.py --docs 25` → PDFs + `questions.parquet`
3. `uv run prag ingest` then `uv run prag stats` → point count == lines in `chunks.jsonl`
4. Running `ingest` **again** leaves the count unchanged (idempotent)
5. `uv run prag search "What was 3M's FY2018 capital expenditure?"` → chunks from the 3M 10-K
6. `uv run pytest` and `uv run ruff check` pass
