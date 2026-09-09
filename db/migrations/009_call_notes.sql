-- Call notes: the call_log fields table (guarded — no synthea schema in CI),
-- and the call-note source becomes live under the provisional employee tier
-- (Phase 2 creates member_services and re-tags it).
DO $$
BEGIN
    IF to_regclass('synthea.patients') IS NOT NULL THEN
        CREATE TABLE IF NOT EXISTS synthea.call_log (
            call_id      text PRIMARY KEY,
            patient      text REFERENCES synthea.patients (id),
            member_id    text NOT NULL,
            call_date    date NOT NULL,
            rep_id       text NOT NULL,
            reason_code  text NOT NULL,
            disposition  text NOT NULL,
            claim_id     text,
            duration_sec integer NOT NULL
        );
    END IF;
END
$$;
UPDATE sources SET acl_tag = 'employee', status = 'ingested' WHERE key = 'call_notes';
UPDATE sources SET status = 'loaded' WHERE key = 'call_log';
-- Near-duplicate documents point at the original and carry no chunks.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS duplicate_of bigint REFERENCES documents (id);
