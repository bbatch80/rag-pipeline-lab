-- One BM25 index per source. BM25's length normalization and IDF are computed
-- over the index: sharing one index across sources with different natural
-- chunk sizes (brochure pages ~980 chars, call notes ~300) measures every
-- source against a blended average and reorders pages within a source for
-- reasons unrelated to the question. Per-source indexes keep each source's
-- statistics its own. Each new vector source's migration adds its index;
-- `raglab index` reindexes them all after bulk loads.
DROP INDEX IF EXISTS chunks_bm25_idx;
CREATE INDEX IF NOT EXISTS chunks_bm25_brochure_idx      ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'brochure';
CREATE INDEX IF NOT EXISTS chunks_bm25_rates_idx         ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'rates';
CREATE INDEX IF NOT EXISTS chunks_bm25_sop_idx           ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'sop';
CREATE INDEX IF NOT EXISTS chunks_bm25_bulletin_idx      ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'bulletin';
CREATE INDEX IF NOT EXISTS chunks_bm25_formulary_idx     ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'formulary';
CREATE INDEX IF NOT EXISTS chunks_bm25_kb_idx            ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'kb';
CREATE INDEX IF NOT EXISTS chunks_bm25_clinical_note_idx ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'clinical_note';
CREATE INDEX IF NOT EXISTS chunks_bm25_call_note_idx     ON chunks USING bm25 (index_text) WITH (text_config = 'english') WHERE doc_type = 'call_note';
