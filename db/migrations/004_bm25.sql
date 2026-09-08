-- Real BM25 for the lexical arm (pg_textsearch). The library is preloaded by
-- the database image; the extension and index are per database. The index
-- scores chunk text with IDF and length normalization, which Postgres's
-- built-in ts_rank_cd lacks — the reason v1 queried rare lexemes only.
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON chunks USING bm25 (content)
    WITH (text_config = 'english');
