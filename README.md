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

## Orchestration

Three Airflow DAGs (`dags/`): `raglab_ingest_sync` (daily freshness SLA —
download, hash-diff ingest, embed, index; content fingerprints cover source
bytes AND processing config, so config changes rebuild exactly the affected
documents), `raglab_eval_drift` (daily: gated retrieval eval + embedding
drift check + dashboard refresh), and `raglab_backfill` (manual: full
re-derive drill for embedding-model swaps). Tasks invoke the CLI commands
1:1 — every task log contains the command's run receipt, and task success is
the receipt's exit code. Verified scenarios: single-document surgical sync
(290 skipped / 1 reingested / 1 embedded), deletion propagation (source gone
→ rows cascade out), full backfill with post-rebuild re-baseline.

Known limitation: Airflow 3.x's task supervisor deadlocks forked task
processes on macOS (apache/airflow#64874, #65691); on macOS dev machines DAG
executions run via Airflow's in-process `dags test` runner. Linux and
containerized deployments use the native executor unaffected.

## Governance

Two lanes, enforcement in the engine — application code never filters
content. Identical questions produce correctly different answers per role,
with an audit trail in both lanes.

### Document lane (Postgres row-level security)

Chunks and documents carry an `acl_tag`; real Postgres roles
(`persona_public` / `persona_employee` / `persona_care_team`) enforce a
lateral need-to-know model under `FORCE ROW LEVEL SECURITY`: everyone sees
public documents, only the employee role sees internal operations content,
only the care_team role sees clinical notes — the two non-public tiers are
mutually invisible. Rows outside a role's entitlement are trimmed by the
engine before ranking, so unauthorized content never enters a candidate
set, a payload, or a context window. Revoking a role membership changes the
retrievable set at the next query with zero re-indexing. HNSW scans run
with `iterative_scan=relaxed_order` so heavily-trimmed roles still fill k
results (tested at 3% row visibility).

Every retrieval flows through one pipeline entrypoint
(`raglab.pipeline.run_query`) that assumes the caller's role via
`SET LOCAL ROLE` and writes a disclosure record: persona, query, payload
id, chunk ids, content hashes, document titles, ACL basis, score.
The disclosure log has no foreign keys and denormalizes document identity,
so audit records survive document deletion and reingest. `raglab audit`
reports both directions: what a persona saw, and which payloads used a
given document.

```sh
uv run raglab query "<question>" --persona care_team   # full context payload JSON
uv run raglab explain "<question>" --persona employee --generate
uv run raglab audit --document "<title>"    # lineage: payloads that used it
uv run raglab audit --persona public        # disclosure: what a role saw
```

### PHI de-identification (Presidio, before indexing)

Clinical notes are de-identified at ingest — before embedding — so
protected text never enters the embedding space, the searchable corpus, or
any model API. Two modes (`RAGLAB_DEID=mask|tokenize`; the mode is part of
the document processing recipe, so flipping it re-ingests exactly the
affected documents). Tokenize mode issues consistent pseudonyms
(`[PERSON-0002]` is the same patient in every note) backed by an owner-only
vault table; persona roles cannot read the vault. Queries from
vault-entitled sessions (admin, care_team) are translated
name → pseudonym before search, restricted to lookup-identifier entity
types, so de-identified notes remain searchable by the identifiers
clinicians actually use — for entitled roles only.

Detection is scored against a generation-time PHI injection manifest
(ground truth by construction), written to the metrics store by
`raglab deid-eval`:

| entity type | detection recall |
|---|---:|
| ssn | 1.000 |
| member_id | 0.984 |
| phone | 0.936 |
| address | 0.926 |
| date | 0.827 |
| name | 0.824 |
| mrn | 0.325 |
| **overall** | **0.854** |

Corpus leakage rate (injected entities surviving verbatim in indexed
text): **12.45%**, dominated by MRNs and shorthand dates — stock Presidio
has no recognizer for bare medical record numbers in clinical shorthand.
Production hardening is custom recognizers for local identifier formats,
re-measured against the same manifest.

### Member-data lane (Snowflake row access policies + dynamic masking)

Synthea claims (11,519 patients, 676,859 claim lines) served from
Snowflake as a governed copy; Postgres remains the system of record.
`raglab snowflake-setup` rebuilds the lane end-to-end (warehouse, roles,
tables, policies, load) idempotently in ~25 s. One `SELECT` against
`CLAIM_DETAIL`, four result shapes — enforced by the warehouse:

| role | rows | identity columns | financial columns |
|---|---:|---|---|
| CLAIMS_EXAMINER | 676,859 | visible | visible |
| PSHB_EXAMINER | 126,799 (PSHB book only) | visible | visible |
| CARE_MANAGER | 676,859 | visible | masked (NULL) |
| ACTUARY | 676,859 | SSN NULL, names → SHA-256, DOB → year | visible |

The actuary's hashes are stable, so de-identified member-level aggregation
still works (`COUNT(DISTINCT ...)` matches the examiner's). Row scope is a
row access policy consulting an owner-only entitlement table; sessions pin
a single role (`USE SECONDARY ROLES NONE`), as a production service
connection would. Audit is Snowflake's own `ACCESS_HISTORY` — consumed,
not built — which also records the policy's entitlement-table lookups.
`raglab snowflake-verify` asserts the full matrix; the same assertions run
as pytest (skipped where no credentials exist — CI holds no secrets).

### Entitlement assertions in CI

The golden set includes persona-negative questions asserting both
directions per entitlement wall: the unauthorized persona must return
`insufficient_evidence` and the authorized persona must answer citing the
expected document, run through the full persona pipeline (RLS, vault
translation, disclosure). `deny_abstained`, `allow_answered`, and
retrieval `hit@5` gate CI; the entitlement metrics are thresholded at 1.0 —
a single leak fails the build. Payload `status`/`confidence` inform the
consumer; grounding is enforced at the generation layer, whose contract
(answer only from supplied chunks, refuse otherwise) is exercised by
abstention-trap questions in the generation eval.
