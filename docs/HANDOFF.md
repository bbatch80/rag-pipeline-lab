# Hand-off specification

What a team that did not build this platform needs to run it, reproduce
any payload it ever delivered, and know that a change did not break it.
Five parts. Every claim points at a file, a table, or a command in this
repository; nothing here is aspirational.

## 1. Payload contract

Every retrieval — a surface, the CLI, the evaluation harness, or an MCP
tool — returns one JSON payload. The normative definition is the JSON
Schema at `db/payload.schema.json`; `raglab.payload.SPEC_VERSION` is the
version the code emits, and `tests/test_payload_spec.py` validates every
shape the builder can produce against the schema in CI.

| | |
|---|---|
| Spec version | `1.2.0` (semver: additive changes bump minor, breaking changes bump major; consumers pin against `spec_version`) |
| Required fields | `spec_version`, `query`, `status`, `retrieved_at`, `chunks` |
| `status` | `ok` · `insufficient_evidence` · `out_of_scope` — decided upstream (abstention threshold, scope gate); the consumer relays it and never invents a refusal or an answer |
| `confidence` | best reranker score, 0–1; informs, never authorizes |
| `boundary_response` | on `out_of_scope` only: the exact text to relay |
| `payload_id` | joins the payload to its disclosure record (part 4) |
| `router` | `years`, `plan_codes`, `as_of`, `plan_from_enrollment`: what the platform searched, not what the model guessed |
| `plan`, `sub_results`, `warehouse_results`, `missing` | the composed shape (one question, several legs): what was looked at, what each leg returned, which required legs returned nothing |
| `coverage` | when a program is named without a plan: every plan searched, and which had evidence |
| `member_context`, `record_context`, `unresolved_identifiers`, `as_of_defaulted`, `subject` | the identifiers the platform bound (member, case, claim, date of service) and the ones it could not |
| `timings`, `search` | per-stage latency and the search parameters in force |

Each entry in `chunks[]` is a self-contained citation unit:

| Field | Meaning |
|---|---|
| `text` | the chunk as delivered (PHI already pseudonymized at ingest) |
| `source` | `title`, `path`, `plan_code`, `year`, `pages`, `section`, `doc_type`, `content_hash` (source bytes + processing recipe), `record` (the governed record fields for member records — never extracted from text) |
| `acl_basis` | the entitlement tier that admitted this chunk for this caller: `public`, `employee`, `care_team`, `member_services`, `appeals` |
| `provenance` | `document_version` (a label read from governed metadata), `recipe` (the processing recipe that produced the document), `embedding_model` (the model that embedded the chunk); `null` where the corpus predates recording — never guessed |
| `scores` | `rrf` (fusion) and `rerank` (cross-encoder) |

**Metadata that feeds retrieval.** Every chunk row carries typed columns
the router and the entitlement policies filter on — `doc_type`, `year`,
`plan_code`, `acl_tag`, `member_key`, `appeal_cited`, `embedding_model` —
and a `metadata` JSON object with `program`, `carrier`, `plan_options`,
`effective_date`, `section`, `pages`, `categories`, and `record` (the
issuing system's required fields: case, claim, call, policy, bulletin,
letter ids, versions, effective windows). Identifiers in a question
resolve into filters on these columns; they are never search words.

**MCP tools** (`raglab.mcp_server`; identity is launch configuration,
never a tool parameter): `compose_context(question, member_id?, case_id?,
module?)` returns the composed payload; `search_documents(query,
member_id?)` the single-probe payload; `query_member_data(query_name,
…)` runs one of the named warehouse queries in
`raglab.snowlane.NAMED_QUERIES` as the session's warehouse role and
names its `masked_columns`. No freeform SQL and no retrieval or
governance logic crosses the tool boundary.

## 2. Sources and recipes

The registry is the `sources` table (seeded by `db/schema.sql` and the
migrations; `raglab.sources` reads it). Eleven document sources feed the
index; the generator and CSV sources feed the warehouse lane.

| Source | Type | Tier | PHI | Versioned | Member-scoped | Event | Chunking | Parser |
|---|---|---|---|---|---|---|---|---|
| brochures | brochure | public | | | | | section | pdf (hi_res layout) |
| rates | rates | public | | | | | row | csv |
| sops | sop | employee | | yes | | | section | markdown |
| bulletins | bulletin | employee | | yes | | | section | markdown |
| formulary | formulary | employee | | yes | | | section | markdown |
| kb | kb | employee | | | | | section | markdown |
| clinical_notes | clinical_note | care_team | yes | | yes | | section | markdown |
| call_notes | call_note | member_services | yes | | yes | yes | record | markdown |
| appeal_documents | appeal | appeals | yes | | yes | yes | section | markdown |
| clinical_policies | clinical_policy | public | | yes | | yes | section | markdown |
| carrier_letters | carrier_letter | public | | | | yes | section | pdf |

*Versioned*: a default question sees the version in effect, a dated
question the version in effect on that date. *Member-scoped*: searched
only inside one member's context. *Event*: dated by issue, exempt from
the plan-year filter.

**The recipe** (`raglab.ingest.processing_recipe`) is the processing half
of a document's identity: parser backend, chunk limits, contextual mode,
the de-identification mode and version for PHI sources, the search-copy
dictionary and boilerplate threshold for normalized sources, and the
record-header marker for record sources. It is folded into every
document's `content_hash` and stored as text in `documents.recipe`, so
a document is stale exactly when its bytes or its recipe change.

**Rebuilding the index** from source files: `raglab ingest --full`
(parse, chunk, gate, de-identify, load; unchanged documents are skipped
by fingerprint), `raglab embed` (embeds where `embedding IS NULL`;
resumable; writes `embedding_model`), `raglab index` (vacuum, then the
HNSW and per-source BM25 indexes). A change to the *derived* search copy
only — `raglab rebuild-search-copy` — never re-de-identifies. Embedding
model: `raglab.embed.MODEL`. Search-time parameters: `raglab.retrieval`
(`EF_SEARCH`, `MAX_SCAN_TUPLES`, fusion and pool sizes). Schema changes
are numbered migrations in `db/migrations/`, applied once each by
`raglab migrate`, which the deployed container runs at every start.

## 3. Entitlement rules

Entitlement is enforced in the database before ranking, so an
unauthorized chunk is never a candidate. Five tiers compose by Postgres
role inheritance (`db/governance.sql`; the current policies are in
`db/migrations/023_appeal_cited_flag.sql`):

- `public` — everyone; `employee` — any staff persona; `care_team`,
  `member_services`, `appeals` — the named persona only;
- the relational branch: a `care_team` or `member_services` document
  cited by an appeal (`appeal_cited`, maintained by trigger from
  `appeal_evidence`) is also visible to `appeals`.

Identity groups (`raglab.identity.GROUPS`) pair a document persona with
a warehouse role and a set of surfaces:

| Group | Document persona | Warehouse role | Surfaces |
|---|---|---|---|
| public | public | — | ask |
| call_center | member_services | MEMBER_SERVICES_REP | ask, agent_assist |
| appeals | appeals | APPEALS_ANALYST | ask, appeals_workbench |
| benefits | employee | — | ask |
| care_management | care_team | CARE_MANAGER | ask, agent_assist |
| analytics | public | ACTUARY (names, city masked; ZIP to 3 digits) | ask, analyst_view |
| admin | owner connection, unscoped | CLAIMS_EXAMINER | all |

Warehouse rows are shaped by Snowflake row-access and masking policies
(`db/snowflake/setup.sql`); every named query runs with `QUERY_TAG =
payload_id`. PHI is pseudonymized before indexing (`raglab.deid`; the
vault `deid_vault` is owner-only and reversible only there). The
persona-negative golden items assert both directions of every wall in
CI, and `raglab recall-rls` measures vector recall per persona so
entitlement never silently costs retrieval quality.

## 4. Reproducibility

Every delivery is a row in `disclosure_log`: who asked (`user_id`,
`persona`), from where (`source`: web, mcp, eval), the question, the
`payload_id`, the status, the chunk ids, the content hashes, the titles,
the entitlement bases, per-stage `timings`, and the **payload itself,
verbatim** (`payload` JSON). The Console renders any row by id.

To reproduce or revoke a logged payload later:

1. `SELECT payload FROM disclosure_log WHERE payload_id = …` — the exact
   context that was delivered, no re-retrieval needed.
2. Each chunk's `source.content_hash` identifies the exact document
   version; `provenance.recipe` and `provenance.embedding_model` say
   whether the corpus that produced it is the corpus you have now
   (compare with `documents.recipe`, `chunks.embedding_model`, and the
   current `processing_recipe`).
3. `acl_basis` per chunk says which entitlement admitted it, so a
   revocation question ("should this role have seen this?") is answered
   from the row, not from memory.
4. Warehouse legs join to Snowflake `QUERY_HISTORY` / `ACCESS_HISTORY`
   on the payload id.

The log has no foreign keys to the corpus on purpose: audit rows outlive
re-ingests and deletions.

## 5. Acceptance

- **Golden set**: `eval/golden.jsonl` (161 questions across lookup,
  guardrail, comparison, temporal, linked, summarization, explanation,
  diagnosis, and quantitative work, each with its expected documents,
  values, rows, or refusal). `raglab eval-retrieval --gate` runs it and
  exits non-zero when any per-item result falls below `eval/baseline.json`
  (a ratchet: nothing that passes may start failing; each guardrail item
  is ratcheted individually).
- **CI** (`.github/workflows/`): the `test` job builds an empty database
  from `db/*.sql` plus every migration and runs the suite; the `gate` job
  runs the golden ratchet beside the loaded corpus. Both are required on
  `main`.
- **Release**: a `v2.*` tag builds the images; a person approves the
  `production` deployment; `deploy/deploy.sh up` starts the new version,
  `raglab smoke` checks the live site as every role, and a failed smoke
  rolls back.
- **Per-persona retrieval quality**: `raglab recall-rls` (recall@50
  under row-level security, per persona, against an exact scan);
  `raglab benchmark` for index parameters; `raglab deid-eval` for PHI
  detection recall and leakage.
