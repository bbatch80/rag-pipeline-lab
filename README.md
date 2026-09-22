# rag-pipeline-lab

https://github.com/user-attachments/assets/e11f2e70-c473-4fc1-97fc-23ba1b1af58a

*The five surfaces answering questions as six roles, with the disclosure
log and the evaluation dashboard behind them (5:52, no audio). Full
quality: [release v2.0.25](https://github.com/bbatch80/rag-pipeline-lab/releases/tag/v2.0.25).
The questions shown were already cached, about a second each; a fresh
question reranks on CPU in about a minute, see [Latency](#latency-honestly).
The site ran at `ragpipelinelab.com` on an Azure VM through September
2026 and is not kept up: the model-API and cloud costs of a public
endpoint are not justified for a portfolio project.*

A governed retrieval pipeline over health-plan documents and member data:
parsing, PHI de-identification, hybrid retrieval with reranking, role-based
access enforced in the database, a versioned context payload served over
MCP and HTTP, an audit log of every payload delivered, and an evaluation
gate on every change.

**All member data is synthetic.** The population comes from
[Synthea](https://github.com/synthetichealth/synthea); every clinical
note, call note, appeal, name, identifier and date is generated about it.
The plan brochures and carrier letters are OPM's public PDFs. No real PHI
exists in this repository or its history.

## What it does

One paragraph per requirement, with the number that backs it.

**Extraction and chunking.** Eighteen sources, each declared once in a
table (lane, parser, chunk profile, gate rules, PHI flag). Brochure PDFs
go through Unstructured `hi_res`, chosen over Docling by a bake-off on a
table-row metric rather than on hit@5 alone; a call note or an appeal is
one record, one chunk. 10,797 documents become 23,589 chunks.

**Healthcare compliance: PHI masked or tokenized before indexing.**
Presidio plus custom recognizers for the identifiers a payer actually
uses (member IDs and claim numbers with check digits, MRNs, roster
names), so protected text never reaches the embedding space, the index,
or a model API. Two modes: mask, or tokenize into consistent pseudonyms
backed by an owner-only vault. Scored type-correct against
generation-time manifests: recall 0.657 with stock Presidio, 0.999 with
the recognizers; 0.20% of entities survive in an indexed chunk. CI gates
recall ≥ 0.98 per identifier type and leakage < 5%. Stated plainly: the
corpus is minimized under HIPAA's identifier list, not de-identified
under Safe Harbor, and Expert Determination is not claimed.

**Metadata and routing.** Every chunk carries plan, year, section, the
plan options a heading names, record keys, a policy version's effective
window, and provenance (document version, processing recipe, embedding
model). An identifier in a question is recognized by shape and check
digit and becomes a filter, never a search term. A pinned model reads the
route (scope, years, options) and the code enforces it; the scope gate
refused 6/6 out-of-domain probes without retrieval.

**Index design, embedding scheme, hybrid search, reranking.** What is
embedded is not what is shown: a record is indexed from a search copy
(identifiers canonicalized, rep shorthand expanded, boilerplate dropped)
and cited verbatim, and every chunk carries a metadata-template context
line that beat model-written context on precision at no cost per
rebuild. pgvector HNSW at an operating point picked by rule (recall
0.996 at 1.2 ms, `ef_search=40`), one BM25 index per source, RRF fusion
in SQL, then Qwen3-Reranker-0.6B as a yes/no judge whose probability is
also the abstention signal (bar 0.7).
bge-reranker-base stays in the image as a one-line fallback. The
reranker delivers the decisive lift: hit@5 0.862 fused, 0.966 reranked
(the full funnel ablation is in the details).

**Context-level access control.** Documents: Postgres row-level security
under `FORCE ROW LEVEL SECURITY`, five tiers by role membership, member
scope applied as a filter before ranking, and one data-driven entitlement
(a note is visible to appeals only while an appeal cites it). Member data:
Snowflake row-access and masking policies across six roles. Identity is
the session, never a request field; a `persona` or `role` in a request
body is a 400. Zero leaks, gated in CI.

**Agentic context services.** One versioned payload (spec 1.2.0): status,
confidence, chunks with source, ACL basis, provenance and per-stage
scores, warehouse rows with masked columns labeled, and a payload id. A
question becomes a plan of up to three legs (document probes and named
warehouse queries), planned by a pinned model, bound to identifiers by the
platform, run as the caller. Served three ways from one implementation:
CLI, an MCP server with three tools, and an HTTP API behind five
server-rendered surfaces (Ask, Agent Assist, Appeals Workbench, Analyst
View, Console).

**Auditing and hand-off lineage.** Every payload is written to a
disclosure log before it is returned, fail-closed: no audit row, no
context. Any payload is reproducible by id; the log answers both what a
role saw and which payloads used a document; Snowflake `QUERY_TAG`
carries the payload id into the warehouse's own access history. The
hand-off to a receiving team is the contract, not the code: a normative
JSON schema validated in CI, semver on the payload, migrations that ship
with the release tag, a snapshot that restores the whole store, and the
production mapping below.

**Retrieval evaluation and groundedness.** A 163-question golden set with
human-verified sources across ten work categories, scored on the checks
each question declares (values, documents, legs, rows, masked columns,
versions kept out, a persona wall). The framework is purpose-built
rather than Ragas or TruLens because the checks are the platform's own
invariants, not generic relevance scores. The CI gate is a ratchet
against a pinned baseline: no category, group or guardrail may fall. Current
baseline: hit@5 0.93, 121 of 163 items passing. A paired diff names every
question that flipped; a dashboard tracks progress by release; a daily
DAG checks embedding drift.

**Orchestration and delivery.** Three Airflow DAGs (ingest-sync with
content fingerprints, eval-drift, backfill) call the same CLI commands,
each ending in a run receipt. The golden paths: protected `main`, one PR
at a time, 368 tests, the eval gate on a self-hosted runner beside the
indexed database.
A git tag is a release: build, reviewer approval, deploy, smoke test, or
rollback.

**Grounded generation and guardrails.** The answer a surface shows is
generated from the payload beneath it and nothing else, by a pinned model
(Claude Haiku 4.5; a second vendor is wired in for cross-checks), and only
when the platform served evidence. Refusals come from the pipeline, not
the model: the abstention bar and the scope gate set the status, and the
boundary text for out-of-scope questions travels in the payload. The
golden set carries 24 guardrail items: persona walls, member scope,
superseded versions, unanswerable questions, and questions that try to
instruct the pipeline ("skip the version checks"), each ratcheted
individually. A generation eval scores correctness and faithfulness with a
cross-family judge and fails when an unanswerable question gets an answer.

**Stack.** Python and SQL · PostgreSQL 17, pgvector, pg_textsearch · Snowflake · Unstructured · Presidio · PyTorch · OpenAI embeddings · Anthropic and OpenAI generation · FastAPI, htmx · MCP · Airflow 3 · Docker, Caddy · GitHub Actions · Azure.

## Latency, honestly

There is no GPU behind this build. On the 8-vCPU Azure VM a fresh
question took 64–75 s, 60–69 s of it the reranker scoring roughly two
hundred candidates on CPU.
A question the platform has seen takes about 1.3 s: plans, query
embeddings and reranker scores are cached in Postgres, keyed by content,
so a repeat costs nothing, a re-ingest misses honestly, and the caches
survive deploys. Accuracy was chosen over latency for this build; a small
GPU brings the rerank stage under half a second.

## Reproduce

```sh
docker compose up -d                 # pgvector + BM25 on port 5433
uv sync
uv run raglab init-db && uv run raglab download && uv run raglab ingest
uv run raglab eval-retrieval --gate  # the golden set against the baseline
uv run raglab serve                  # the surfaces at http://127.0.0.1:8000
```

## Production mapping

| lab | production analog |
|---|---|
| pgvector + pg_textsearch (hybrid retrieval, RLS) | Azure AI Search with security trimming |
| Postgres roles / RLS personas | Entra ID app roles resolved server-side |
| Snowflake row access + masking policies | Snowflake (same) |
| Airflow DAGs | Azure Data Factory |
| GitHub Actions + self-hosted runner beside the database | Azure DevOps Pipelines + self-hosted agent in the VNet |
| Release-tagged Compose deploy | Container deploy with an environment approval gate |

The measurements, rules, and rejected alternatives behind every line
above are in [docs/DETAILS.md](docs/DETAILS.md).
