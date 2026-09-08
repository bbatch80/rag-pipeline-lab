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
