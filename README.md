# ProductionRAG

Production-grade RAG over FinanceBench, built iteratively. See `docs/iter-1.md`.

```bash
cp .env.example .env   # add your OPENAI_API_KEY
docker compose up -d   # Qdrant on :6333
uv sync
```
