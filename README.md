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

## Retrieval funnel ablation

Query → rule-based router (scope gate, year resolution, plan filters) →
hybrid search (pgvector + Postgres full-text) → reciprocal rank fusion
(k=60, rank-only, in SQL) → top-50 → cross-encoder reranker
(BAAI/bge-reranker-base, local). Measured as hit@5 on a 29-question golden
set with human-verified source labels (`eval/golden.jsonl`); reproduce with
`raglab ablation`, inspect any query with `raglab explain "<query>"`.

| arm | hit@5 |
|---|---:|
| vector only | 0.793 |
| lexical only | 0.207 |
| RRF fusion | 0.793 |
| RRF + rerank | **0.931** |

Findings: the reranker delivers the decisive lift; fusion ties vector on
natural-language questions and wins on identifier-style queries (exact
member IDs, dollar amounts, enrollment codes hit rank 1 via the lexical
arm — Postgres FTS lacks IDF, so the lexical arm queries rare lexemes only,
using a document-frequency table rebuilt at index time). The router's scope
gate resolved 6/6 out-of-domain/out-of-year probes without retrieval.
Abstention threshold (0.5) separates answerable questions (best rerank
score ≥ 0.72 across the golden set) from absent-topic questions (0.30);
redirect-style questions score high on genuinely-relevant-but-non-answering
chunks and are handled at the generation layer instead.

## Evaluation experiments

Deterministic retrieval suite (`raglab eval-retrieval`) over the golden set,
writing to a metrics store; discrimination-checked (a sabotaged retriever
trips the gate thresholds). Rebuild equivalence verified: wipe + re-ingest
reproduces baseline metrics exactly.

| experiment | arm | hit@5 | yoy coverage | conclusion |
|---|---|---:|---:|---|
| chunk size | small (1200/900/150) | 0.793 | 0.736 | fragments answers; trips gate |
| chunk size | **baseline (2000/1500/250)** | 0.931 | 0.417 | **kept** |
| chunk size | large (3000/2400/400) | 0.862 | 0.799* | overall regression |
| parser bake-off | Unstructured-fast | 0.931 | — | **production default** |
| parser bake-off | Docling (table docs) | 0.862 | — | hypothesis rejected: lost table hit@5 1.0→0.875 |
| year routing | blended search | 0.931 | 0.417 | one year crowds out the other |
| year routing | **per-year search + stratified rerank + router vocab** | **0.966** | **0.521** | metadata-native fix, $0 |
| contextual chunks | model-written chunk context | 0.966 | 0.750 | ties hit@5; large yoy-coverage gain; costs ~$0.70 + hours per rebuild; traded away a factual hit |
| contextual chunks | **metadata-template context** | **0.966** | **0.688** | **adopted as ingest default**: ~80% of the LLM arm's coverage gain, best precision, no noise cost, $0 |

\* overall source_coverage. Multi-year questions are searched per routed
year and re-ranked with per-year slot guarantees — one blended ranking lets
near-identical cross-year chunks crowd each other out of the pool entirely.
