-- Provenance (payload spec 1.2.0): what a hand-off team needs to reproduce
-- or revoke a logged payload later. The recipe was always folded into the
-- document fingerprint but never stored as text; the embedding model was a
-- constant in code. Both become columns. documents.recipe is backfilled on
-- the ingest skip path (a matching fingerprint proves the recipe); the
-- embedding model is backfilled here because exactly one model has ever
-- embedded this corpus.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS recipe text;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model text;
UPDATE chunks SET embedding_model = 'text-embedding-3-small'
 WHERE embedding IS NOT NULL AND embedding_model IS NULL;
