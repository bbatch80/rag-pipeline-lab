-- index_text: what gets embedded and BM25-indexed (the search copy);
-- content stays the verbatim display copy that is cited, shown, and stored
-- in the audit row. Identical for every source until call notes (Phase 1)
-- write a normalized search copy. The BM25 index moves onto it.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS index_text text;
UPDATE chunks SET index_text = content WHERE index_text IS NULL;
ALTER TABLE chunks ALTER COLUMN index_text SET NOT NULL;
-- On an existing database run `raglab index` after this migration: the
-- UPDATE above leaves dead row versions that an index build in the same
-- transaction counts in its statistics (avg length, document count).
DROP INDEX IF EXISTS chunks_bm25_idx;
CREATE INDEX chunks_bm25_idx ON chunks USING bm25 (index_text)
    WITH (text_config = 'english');

-- member_key: the person key (Synthea patient UUID) of member-scoped
-- documents; NULL for sources that are not about one member. Filled by
-- ingest from each source's manifest.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS member_key uuid;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS member_key uuid;
CREATE INDEX IF NOT EXISTS chunks_member_key_idx ON chunks (member_key);
