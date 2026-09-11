-- The registry DECLARES each source's metadata hierarchy (Phase 3.5, the
-- router principle): the router walks it — the deepest level a question names
-- is bound, the level beneath it is covered (one search per key, the best
-- evidence per key). One rule; a new axis is a registry line, not code.
-- Levels name chunk metadata: 'program' derives from plan_code (FEHB/PSHB
-- code sets), 'plan_code' and 'year' are columns, 'policy_id'/'version' live
-- in the record metadata.
ALTER TABLE sources ADD COLUMN IF NOT EXISTS hierarchy text[] NOT NULL DEFAULT '{}';
UPDATE sources SET hierarchy = '{program,plan_code,year}' WHERE key = 'brochures';
UPDATE sources SET hierarchy = '{year}'                   WHERE key IN ('rates', 'carrier_letters');
UPDATE sources SET hierarchy = '{policy_id,version}'      WHERE key = 'clinical_policies';
