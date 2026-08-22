-- raglab schema. Applied on first container boot via docker-entrypoint-initdb.d
-- and on demand via `raglab init-db` (drop-and-recreate).

CREATE EXTENSION IF NOT EXISTS vector;

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
    metadata    jsonb   NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(1536),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Full-text search index. No vector index yet: exact scan is the retrieval
-- baseline until the index benchmark selects an operating point.
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_document_id_idx ON chunks (document_id);
