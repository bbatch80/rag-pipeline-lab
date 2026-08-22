# rag-pipeline-lab

Governed retrieval pipeline over health-plan documents and member data: document parsing, hybrid retrieval, evaluation, orchestration, and access-controlled context services.

## Quickstart

```sh
docker compose up -d      # pgvector (PostgreSQL 17) on port 5433
uv sync
uv run raglab init-db
uv run raglab status
uv run pytest
```
