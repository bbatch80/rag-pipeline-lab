-- Relational entitlement stored on the row (Phase 2 decision 7). The two
-- correlated EXISTS branches (migrations 019, 021) ran as a sub-plan per
-- candidate row — ~275k evidence lookups per question across the per-source
-- queries; every persona call took 13 s. The relationship is now a flag on
-- documents and chunks, maintained by triggers: appeal_evidence stays the
-- source of truth, a citation write flips the flag in the same transaction,
-- and a document or chunk inserted later (re-ingest) picks the flag up at
-- insert. Kind is matched to tier: clinical_note <-> care_team,
-- call_note <-> member_services. The policy branch is then a column read.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS appeal_cited boolean NOT NULL DEFAULT false;
ALTER TABLE chunks    ADD COLUMN IF NOT EXISTS appeal_cited boolean NOT NULL DEFAULT false;

CREATE OR REPLACE FUNCTION appeal_cited_for(p_title text, p_acl_tag text) RETURNS boolean
LANGUAGE sql STABLE AS $$
    SELECT EXISTS (
        SELECT 1 FROM appeal_evidence e
        WHERE e.document_title = p_title
          AND e.kind = CASE p_acl_tag WHEN 'care_team' THEN 'clinical_note'
                                      WHEN 'member_services' THEN 'call_note' END)
$$;

-- A citation changes -> recompute the flag for the title(s) it names.
CREATE OR REPLACE FUNCTION appeal_evidence_sync_flag() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY_REMOVE(ARRAY[
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN NEW.document_title END,
        CASE WHEN TG_OP IN ('DELETE', 'UPDATE') THEN OLD.document_title END], NULL)
    LOOP
        UPDATE documents SET appeal_cited = appeal_cited_for(title, acl_tag) WHERE title = t;
        UPDATE chunks c SET appeal_cited = d.appeal_cited FROM documents d
            WHERE d.id = c.document_id AND d.title = t AND c.appeal_cited IS DISTINCT FROM d.appeal_cited;
    END LOOP;
    RETURN NULL;
END $$;
DROP TRIGGER IF EXISTS appeal_evidence_sync_flag ON appeal_evidence;
CREATE TRIGGER appeal_evidence_sync_flag AFTER INSERT OR UPDATE OR DELETE ON appeal_evidence
    FOR EACH ROW EXECUTE FUNCTION appeal_evidence_sync_flag();

-- A document arrives (ingest, re-ingest) -> its flag comes from the evidence table.
CREATE OR REPLACE FUNCTION documents_set_appeal_cited() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.appeal_cited := appeal_cited_for(NEW.title, NEW.acl_tag);
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS documents_set_appeal_cited ON documents;
CREATE TRIGGER documents_set_appeal_cited BEFORE INSERT OR UPDATE OF title, acl_tag ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_set_appeal_cited();

-- A chunk arrives -> it inherits its document's flag.
CREATE OR REPLACE FUNCTION chunks_set_appeal_cited() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    SELECT d.appeal_cited INTO NEW.appeal_cited FROM documents d WHERE d.id = NEW.document_id;
    NEW.appeal_cited := COALESCE(NEW.appeal_cited, false);
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS chunks_set_appeal_cited ON chunks;
CREATE TRIGGER chunks_set_appeal_cited BEFORE INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_set_appeal_cited();

-- Backfill from the source of truth.
UPDATE documents SET appeal_cited = appeal_cited_for(title, acl_tag);
UPDATE chunks c SET appeal_cited = d.appeal_cited FROM documents d WHERE d.id = c.document_id AND c.appeal_cited IS DISTINCT FROM d.appeal_cited;

-- The policies: the relational branch is a column read.
DROP POLICY IF EXISTS documents_lateral_acl ON documents;
CREATE POLICY documents_lateral_acl ON documents FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'         AND pg_has_role(current_user, 'persona_appeals', 'member'))
    OR (acl_tag IN ('care_team', 'member_services') AND appeal_cited
        AND pg_has_role(current_user, 'persona_appeals', 'member'))
);
DROP POLICY IF EXISTS chunks_lateral_acl ON chunks;
CREATE POLICY chunks_lateral_acl ON chunks FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'         AND pg_has_role(current_user, 'persona_appeals', 'member'))
    OR (acl_tag IN ('care_team', 'member_services') AND appeal_cited
        AND pg_has_role(current_user, 'persona_appeals', 'member'))
);
