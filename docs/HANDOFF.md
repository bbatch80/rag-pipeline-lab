# Hand-off specification

What a team that did not build this platform needs to consume its
context, run it, reproduce any payload it ever delivered, and know that a
change did not break it. Five parts. The tables are generated from the
code when this page is served (`/console/handoff` on the platform); in
the repository they appear as markers. Every claim points at a file, a
table, or a command; nothing here is aspirational.

## 1. Payload contract

Every retrieval — a surface, the CLI, the evaluation harness, or an MCP
tool — returns one JSON payload. The normative definition is the JSON
Schema at `db/payload.schema.json`; the code emits
`raglab.payload.SPEC_VERSION`, and `tests/test_payload_spec.py` validates
every shape the builder can produce against the schema in CI. Semver:
additive changes bump the minor version, breaking changes the major;
consumers pin against `spec_version` and validate against the schema in
their own CI.

Three rules for a consumer:

1. **Honor the status.** `ok` means evidence was served; on
   `insufficient_evidence` say what could not be answered (the composed
   shape names the legs in `missing`); on `out_of_scope` relay
   `boundary_response` verbatim. The consumer never invents a refusal or
   an answer.
2. **Answer only from the payload.** Cite each fact's chunk by source
   title and pages; report warehouse `masked_columns` as "not visible to
   your role".
3. **Keep the `payload_id`.** It joins to the disclosure log; a question
   about a delivered answer is answered from the row, not from memory.

<!-- generated: payload_fields -->

<!-- generated: example_payload -->

### Metadata that feeds retrieval

Every chunk row carries typed columns the router and the entitlement
policies filter on — `doc_type`, `year`, `plan_code`, `acl_tag`,
`member_key`, `appeal_cited`, `embedding_model` — and a `metadata` JSON
object with `program`, `carrier`, `plan_options`, `effective_date`,
`section`, `pages`, `categories`, and `record` (the issuing system's
required fields: case, claim, call, policy, bulletin, and letter ids,
versions, effective windows). Identifiers in a question resolve into
filters on these columns; they are never search words.

### MCP tools

Identity is launch configuration for the server (`RAGLAB_USER`), never a
tool parameter: a model cannot claim an identity. No freeform SQL and no
retrieval or governance logic crosses the tool boundary; the named
warehouse queries are `raglab.snowlane.NAMED_QUERIES`.

<!-- generated: mcp_tools -->

## 2. Sources and recipes

The registry is the `sources` table (`raglab.sources` reads it). The
document sources below feed the index; generator and CSV sources feed
the warehouse lane. *Versioned*: a default question sees the version in
effect, a dated question the version in effect on that date.
*Member-scoped*: searched only inside one member's context. *Event*:
dated by issue, exempt from the plan-year filter.

<!-- generated: sources -->

**The recipe** (`raglab.ingest.processing_recipe`) is the processing half
of a document's identity: parser backend, chunk limits, contextual mode,
the de-identification mode and version for PHI sources, the search-copy
dictionary and boilerplate threshold for normalized sources, and the
record-header marker for record sources. It is folded into every
document's `content_hash` and stored as text in `documents.recipe`, so a
document is stale exactly when its bytes or its recipe change.

**Rebuilding the index** from source files: `raglab ingest --full`
(parse, chunk, gate, de-identify, load; unchanged documents are skipped
by fingerprint), `raglab embed` (embeds where `embedding IS NULL`;
resumable; records `embedding_model`), `raglab index` (vacuum, then the
HNSW and per-source BM25 indexes). A change to the *derived* search copy
only — `raglab rebuild-search-copy` — never re-de-identifies. Embedding
model: `raglab.embed.MODEL`. Search-time parameters: `raglab.retrieval`
(`EF_SEARCH`, `MAX_SCAN_TUPLES`, fusion and pool sizes). Schema changes
are numbered migrations in `db/migrations/`, applied once each by
`raglab migrate`, which the deployed container runs at every start.

## 3. Entitlement rules

Entitlement is enforced in the database before ranking, so an
unauthorized chunk is never a candidate. Five tiers compose by Postgres
role inheritance (`db/governance.sql`; the policies in force are in
`db/migrations/023_appeal_cited_flag.sql`):

- `public` — everyone; `employee` — any staff persona; `care_team`,
  `member_services`, `appeals` — the named persona only;
- the relational branch: a `care_team` or `member_services` document
  cited by an appeal (`appeal_cited`, maintained by trigger from
  `appeal_evidence`) is also visible to `appeals`.

Identity groups pair a document persona with a warehouse role and a set
of surfaces:

<!-- generated: groups -->

Warehouse rows are shaped by Snowflake row-access and masking policies
(`db/snowflake/setup.sql`; the actuary sees names and city masked and ZIP
to three digits); every named query runs with `QUERY_TAG = payload_id`.
PHI is pseudonymized before indexing (`raglab.deid`; the vault
`deid_vault` is owner-only and reversible only there). Persona-negative
golden items assert both directions of every wall in CI, and
`raglab recall-rls` measures vector recall per persona so entitlement
never silently costs retrieval quality.

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
3. `acl_basis` per chunk says which entitlement admitted it, so
   "should this role have seen this?" is answered from the row.
4. Warehouse legs join to Snowflake `QUERY_HISTORY` / `ACCESS_HISTORY`
   on the payload id.

The log has no foreign keys to the corpus on purpose: audit rows outlive
re-ingests and deletions.

## 5. Acceptance

- **Golden set**: `eval/golden.jsonl` — questions across lookup,
  guardrail, comparison, temporal, linked, summarization, explanation,
  diagnosis, and quantitative work, each with its expected documents,
  values, rows, or refusal. `raglab eval-retrieval --gate` runs it and
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
  detection recall and leakage; `raglab drift` for embedding drift on a
  stable chunk sample.
