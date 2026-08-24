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

## Vector index benchmark

HNSW (`m=16`, `ef_construction=64`, cosine) vs exact scan. Corpus: 11,022
embedded chunks (`text-embedding-3-small`, 1536d), 100 sampled queries.
Recall is distance-based: the corpus contains exact-duplicate embeddings
(boilerplate repeated across plan years), so tied neighbors with different
ids are equally correct. Reproduce with `raglab benchmark`.

| ef_search | recall@10 | median latency (ms) |
|---:|---:|---:|
| 10 | 0.996 | 1.9 |
| 20 | 0.996 | 1.7 |
| 40 | 0.996 | 1.2 |
| 80 | 0.996 | 1.4 |
| 120 | 0.996 | 1.7 |
| 200 | 0.996 | 2.1 |

Exact scan median: 42 ms. **Operating point: `ef_search=40`** — recall 0.996
at 1.2 ms (35x speedup) — selected by rule: recall ≥ 0.95 at the lowest
latency achieving it. At this corpus size the index is near-saturated at
minimal search effort; the sweep demonstrates methodology, not necessity.
