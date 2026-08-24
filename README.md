# rag-pipeline-lab

Governed retrieval pipeline over health-plan documents and member data: document parsing, hybrid retrieval, evaluation, orchestration, and access-controlled context services.

## Quickstart

```sh
docker compose up -d      # pgvector (PostgreSQL 17) on port 5433
uv sync
uv run raglab init-db
uv run raglab download    # fetch OPM brochure corpus (--full for all years)
uv run raglab ingest      # parse, chunk, gate, load (--full for all years)
uv run raglab status
uv run pytest
```

The Docling parser backend is optional locally: `uv sync --group docling`,
then `uv run docling-tools models download` (one-time, explicit model fetch).
