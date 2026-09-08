-- v1's lexical arm (ts_rank_cd over a generated tsvector) is gone; BM25
-- (migration 004) is the only lexical arm. Drop its column and index.
DROP INDEX IF EXISTS chunks_tsv_idx;
ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;
DROP TABLE IF EXISTS lexeme_df;
