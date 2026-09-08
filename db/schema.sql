-- raglab schema. Applied on first container boot via docker-entrypoint-initdb.d
-- and on demand via `raglab init-db` (drop-and-recreate).

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS quarantine;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path  text        NOT NULL UNIQUE,
    title        text        NOT NULL,
    content_hash text        NOT NULL,
    acl_tag      text        NOT NULL DEFAULT 'public',
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id bigint  NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content     text    NOT NULL,
    -- Load-bearing fields the engine filters and secures on: typed columns.
    -- Descriptive metadata (carrier, plan_option, doc_type, section, ...): JSONB.
    year        integer,
    plan_code   text,
    acl_tag     text    NOT NULL DEFAULT 'public',
    metadata    jsonb   NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(1536),
    UNIQUE (document_id, chunk_index)
);

-- The BM25 index (pg_textsearch) is created by migration 004 and the HNSW
-- vector index is built by `raglab index`
-- (bulk-load-then-index) with pinned parameters; additive changes to this
-- schema live in db/migrations/ and are applied by `raglab migrate`.
CREATE INDEX chunks_document_id_idx ON chunks (document_id);
CREATE INDEX chunks_year_plan_idx ON chunks (year, plan_code);
CREATE INDEX chunks_acl_tag_idx ON chunks (acl_tag);

-- Documents that failed the ingest quality gate. Never silently skipped:
-- rows here are surfaced by `raglab status` until resolved.
CREATE TABLE quarantine (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path   text        NOT NULL,
    gate          text        NOT NULL,
    detail        text        NOT NULL DEFAULT '',
    quarantined_at timestamptz NOT NULL DEFAULT now()
);
