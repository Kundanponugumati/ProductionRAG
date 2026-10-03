# Iteration 2 — Retrieval evaluation (baseline)

> Measure before fixing. Retrieval eval needs no LLM and costs $0, so it comes before generation.
> (Roadmap change: eval moved ahead of generation.)

## What we measure

For each of the 84 FinanceBench questions: embed the question → top-10 chunks from Qdrant → compare with the labelled evidence page.

| Metric | Question it answers |
|---|---|
| Hit@k (1, 3, 5, 10) | Is an evidence-page chunk in the top k? |
| MRR | How high is the first evidence-page chunk? (1/rank, 0 if not in top 10) |
| Doc Hit@5 | Is the right **document** in the top 5? (catches wrong company / year) |
| Answer-in-context@5 | Does the reference number appear in the top-5 text? (single-number answers only) |

## Code

| File | Role |
|---|---|
| `eval/dataset.py` | `questions.parquet` → `EvalQuestion` (question, answer, doc, evidence pages) |
| `eval/retrieval.py` | `evaluate_retrieval()`, `score_question()`, `summarize()`, `save_run()` |
| `cli.py` | `uv run prag eval-retrieval --name <label>` |

Each run → `data/eval/runs/<timestamp>_<name>/` (`results.jsonl` per question + `summary.json`), and one line appended to `data/eval/history.jsonl` for comparing runs.

## Baseline (iter-1 pipeline: PyMuPDF text, 500-token page chunks, bge-large, dense only)

| hit@1 | hit@3 | hit@5 | hit@10 | MRR | doc_hit@5 |
|---|---|---|---|---|---|
| 0.179 | 0.393 | **0.488** | 0.560 | 0.296 | 0.917 |

| Question type | n | hit@5 | MRR | doc_hit@5 |
|---|---|---|---|---|
| metrics-generated | 11 | 0.82 | 0.59 | 0.91 |
| domain-relevant | 42 | 0.45 | 0.28 | 0.93 |
| novel-generated | 31 | 0.42 | 0.22 | 0.90 |

## Failure analysis

| Outcome | Questions |
|---|---|
| Evidence page in top 5 | 41 (49%) |
| Right doc, wrong page | **36 (43%)** |
| Wrong doc entirely | 7 (8%) |

Findings:
1. **The biggest problem is the page within the document**, not the document. 36 questions reach the right filing but miss the page. Suspects: broken tables, chunks with no context (no company, year or section name in the text), and generic finance wording that matches many pages.
2. **Wrong-document cases are about company and year.** Pfizer 2023 Q2 got Pfizer 2021 ×5. JnJ 10-K and JnJ 8-K got confused in both directions. The JnJ 8-K question got Pfizer. Fix: metadata filters / query understanding.
3. In 9 of the page misses, the top-1 result is a **same-company, different-year** filing.
4. **A score threshold wouldn't help:** the mean top-1 score is 0.723 for hits vs 0.708 for misses.
5. **Answer-in-context isn't useful on FinanceBench:** 10 of the 12 single-number answers are *computed* (ratios, growth %) and never appear in the filing. Only `$1577.00` (3M capex) is copied straight from the filing, and there it does what it should: the labelled page was missed, but the answer was retrieved. The metric is still saved per run, but it isn't a headline number.

## Candidate fixes (one at a time, re-run eval after each)

| Fix | Targets |
|---|---|
| Contextual chunk header (company, year, doc type, page in the embedded text) | wrong year/doc + context-less chunks |
| Metadata filtering by company/year from the query | wrong doc |
| Table-aware parsing | right doc, wrong page (financial statements) |
| Chunk size / structure-aware chunking | right doc, wrong page |
| Hybrid (BM25 + dense) + reranker | generic-wording misses |
