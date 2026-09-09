-- A near-duplicate follows its original: replacing the original removes its
-- copies' rows, and the next ingest re-links them. (A dangling copy would be
-- neither original nor duplicate.)
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_duplicate_of_fkey;
ALTER TABLE documents ADD CONSTRAINT documents_duplicate_of_fkey
    FOREIGN KEY (duplicate_of) REFERENCES documents (id) ON DELETE CASCADE;
