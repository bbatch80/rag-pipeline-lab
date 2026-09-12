-- Eval metrics store. Additive — NEVER drop-and-recreate: metric history is
-- longitudinal data (the drift DAG reads it). Applied idempotently by the
-- eval commands themselves.

CREATE TABLE IF NOT EXISTS eval_runs (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at   timestamptz NOT NULL DEFAULT now(),
    kind         text NOT NULL,               -- retrieval | generation
    config_label text NOT NULL DEFAULT 'baseline',
    git_sha      text,
    notes        text
);

CREATE TABLE IF NOT EXISTS eval_scores (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id      bigint NOT NULL REFERENCES eval_runs (id),
    question_id text NOT NULL,
    category    text NOT NULL,
    metric      text NOT NULL,
    value       numeric NOT NULL,
    generator   text,
    judge       text,
    detail      jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS eval_scores_run_idx ON eval_scores (run_id);
CREATE INDEX IF NOT EXISTS eval_scores_metric_idx ON eval_scores (metric);

-- Every run names the corpus it measured: a digest over the documents'
-- content hashes, so a number can be traced to a corpus version the same
-- way git_sha traces it to a code version.
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS corpus_hash text;

-- Query-embedding cache: the golden questions barely change between runs,
-- and the embed stage is a paid network call per question. Keyed on the
-- exact text; the eval and the pipeline read it before calling the API.
-- RAGLAB_EMBED_CACHE=off bypasses it (tests, A/Bs of the embedder).
CREATE TABLE IF NOT EXISTS query_embeddings (
    model      text NOT NULL,
    text_hash  text NOT NULL,
    embedding  text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, text_hash)
);

-- Rerank score cache: the cross-encoder is a pure function of (model,
-- query text, chunk text), so a score is memoized on exactly those three —
-- the model name plus the weight snapshot it loaded, the final ranking
-- query string, and the sha256 of the exact text scored. No chunk id, no
-- timestamp: any change to what the reranker reads changes the key.
-- RAGLAB_RERANK_CACHE=off bypasses it (tests patch the model).
CREATE TABLE IF NOT EXISTS rerank_scores (
    model      text NOT NULL,
    query_hash text NOT NULL,
    text_hash  text NOT NULL,
    score      real NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, query_hash, text_hash)
);

-- Stored plans (Phase 3 decision 3): a planner model's decision for a
-- question is memoized on exactly what produced it — the sha256 of the
-- TRANSLATED question (vault tokens, never raw identifiers), the sha256 of
-- the menu it chose from (source families + named-query catalog), and the
-- pinned model id. Edit the question, change the menu, or re-pin the model
-- and the key changes; otherwise the CI gate reads the stored plan and never
-- calls the model. The table doubles as the planner's decision log (decision
-- 1: shape per question feeds the classifier trigger). RAGLAB_PLAN_CACHE=off
-- bypasses it.
CREATE TABLE IF NOT EXISTS plans (
    question_hash   text NOT NULL,
    menu_hash       text NOT NULL,
    model           text NOT NULL,
    question        text NOT NULL,          -- translated form: tokens, no PHI
    shape           text NOT NULL,          -- simple | compound
    origin          text NOT NULL,          -- model | rules (fallback)
    plan            jsonb NOT NULL,
    fallback_reason text,
    latency_ms      real,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (question_hash, menu_hash, model)
);

-- Parser bake-off scratch (Phase 3.5, 2026-09-12): every parser's chunks
-- for the same documents, side by side, so a comparison stays open while
-- the golden set is rebuilt. Never read by the pipeline; nothing here is
-- embedded or searched. Promotion of a winner is a separate, backed-up step.
CREATE TABLE IF NOT EXISTS chunks_bakeoff (
    parser       text NOT NULL,
    source_path  text NOT NULL,
    chunk_index  int  NOT NULL,
    section      text NOT NULL DEFAULT '',
    pages        int[] NOT NULL DEFAULT '{}',
    categories   text[] NOT NULL DEFAULT '{}',
    content      text NOT NULL,
    parse_ms     real,
    PRIMARY KEY (parser, source_path, chunk_index)
);
