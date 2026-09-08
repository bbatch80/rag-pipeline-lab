-- doc_type becomes a typed, indexed column on chunks, and every document is
-- bound to its source row. Backfilled from the JSONB copy for existing rows.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_type text;
UPDATE chunks SET doc_type = metadata->>'doc_type' WHERE doc_type IS NULL;
ALTER TABLE chunks ALTER COLUMN doc_type SET NOT NULL;
CREATE INDEX IF NOT EXISTS chunks_doc_type_idx ON chunks (doc_type);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id smallint REFERENCES sources (source_id);
UPDATE documents d
SET source_id = s.source_id
FROM (SELECT DISTINCT document_id, doc_type FROM chunks) c
JOIN sources s ON s.doc_type = c.doc_type
WHERE d.id = c.document_id AND d.source_id IS NULL;
ALTER TABLE documents ALTER COLUMN source_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS documents_source_id_idx ON documents (source_id);
