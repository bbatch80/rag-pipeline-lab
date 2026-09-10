-- Governance layer: lateral need-to-know RLS + disclosure log.
-- Applied by `raglab init-db` after schema.sql (policies attach to the
-- recreated tables; roles are cluster-level and persist).
--
-- Access model (HIPAA minimum-necessary):
--   public    -> acl_tag = 'public' only
--   employee  -> public + employee (SOPs, bulletins, formulary, KB)
--   care_team -> public + care_team (clinical notes)
--   member_services -> public + employee + call notes (inherits persona_employee)
--   appeals         -> public + employee + appeal documents (inherits persona_employee)
-- The clinical tier and the operations tiers never see each other. The engine filters BEFORE ranking:
-- authorization happens before content can enter any context window.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_public') THEN
        CREATE ROLE persona_public NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_employee') THEN
        CREATE ROLE persona_employee NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_care_team') THEN
        CREATE ROLE persona_care_team NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_member_services') THEN
        CREATE ROLE persona_member_services NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_appeals') THEN
        CREATE ROLE persona_appeals NOLOGIN;
    END IF;
END
$$;
-- Tiers compose by inheritance (Phase 2 decision 1).
GRANT persona_employee TO persona_member_services, persona_appeals;

-- The app user may assume any persona (SET ROLE); with no persona set it is
-- a member of all three, i.e. the administrative full view.
GRANT persona_public, persona_employee, persona_care_team, persona_member_services, persona_appeals TO raglab;

GRANT SELECT ON documents, chunks TO
    persona_public, persona_employee, persona_care_team, persona_member_services, persona_appeals;

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

-- The relational branch below reads appeal_evidence; on a fresh database this
-- file runs before migration 013 creates it, so the same definition lives
-- here (idempotent both ways).
CREATE TABLE IF NOT EXISTS appeal_evidence (
    case_id        text NOT NULL,
    document_title text NOT NULL,
    kind           text NOT NULL,
    PRIMARY KEY (case_id, document_title)
);
CREATE INDEX IF NOT EXISTS appeal_evidence_document_idx ON appeal_evidence (document_title);

-- Relational entitlement stored on the row (Phase 2 decision 7): a flag on
-- documents and chunks, maintained from appeal_evidence by triggers (migration
-- 023 carries the same definitions; both are idempotent). The policy reads
-- the column — no per-row sub-plan.
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
CREATE OR REPLACE FUNCTION documents_set_appeal_cited() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.appeal_cited := appeal_cited_for(NEW.title, NEW.acl_tag);
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS documents_set_appeal_cited ON documents;
CREATE TRIGGER documents_set_appeal_cited BEFORE INSERT OR UPDATE OF title, acl_tag ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_set_appeal_cited();
CREATE OR REPLACE FUNCTION chunks_set_appeal_cited() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    SELECT d.appeal_cited INTO NEW.appeal_cited FROM documents d WHERE d.id = NEW.document_id;
    NEW.appeal_cited := COALESCE(NEW.appeal_cited, false);
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS chunks_set_appeal_cited ON chunks;
CREATE TRIGGER chunks_set_appeal_cited BEFORE INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_set_appeal_cited();

DROP POLICY IF EXISTS chunks_lateral_acl ON chunks;
CREATE POLICY chunks_lateral_acl ON chunks FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'
        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'
        AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services'
        AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'
        AND pg_has_role(current_user, 'persona_appeals', 'member'))
    -- relational (Phase 2 decisions 2, 6, 7): a clinical note or call note an
    -- appeal cites — stored on the row as appeal_cited, kept true by triggers
    OR (acl_tag IN ('care_team', 'member_services') AND appeal_cited
        AND pg_has_role(current_user, 'persona_appeals', 'member'))
);

DROP POLICY IF EXISTS documents_lateral_acl ON documents;
CREATE POLICY documents_lateral_acl ON documents FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'
        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'
        AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services'
        AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'
        AND pg_has_role(current_user, 'persona_appeals', 'member'))
    -- relational (Phase 2 decisions 2, 6, 7): a clinical note or call note an
    -- appeal cites — stored on the row as appeal_cited, kept true by triggers
    OR (acl_tag IN ('care_team', 'member_services') AND appeal_cited
        AND pg_has_role(current_user, 'persona_appeals', 'member'))
);

-- Writes stay owner-only: personas are read-only consumers.
GRANT SELECT ON appeal_evidence TO persona_public, persona_employee, persona_care_team, persona_member_services, persona_appeals;
CREATE INDEX IF NOT EXISTS documents_title_idx ON documents (title);

-- Disclosure log: who asked, what was returned, on what authority.
-- DELIBERATELY NO FOREIGN KEYS — audit records must not share the corpus
-- lifecycle. Chunk ids + content hashes + titles are denormalized so the
-- record survives document deletion and reingest.
CREATE TABLE IF NOT EXISTS disclosure_log (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    asked_at       timestamptz NOT NULL DEFAULT now(),
    persona        text NOT NULL,
    source         text NOT NULL,               -- interactive | eval | mcp
    query          text NOT NULL,
    payload_id     uuid NOT NULL,
    payload_status text NOT NULL,
    chunk_ids      bigint[] NOT NULL,
    content_hashes text[] NOT NULL,
    doc_titles     text[] NOT NULL,
    acl_basis      text[] NOT NULL,
    top_score      numeric
);

CREATE INDEX IF NOT EXISTS disclosure_log_asked_idx ON disclosure_log (asked_at);

-- The disclosure record carries the EXACT payload delivered (spec JSON):
-- "who saw what" means the verbatim context, reproducible by payload_id,
-- not just which titles.
ALTER TABLE disclosure_log ADD COLUMN IF NOT EXISTS payload jsonb;

-- Tokenization vault: original PHI recoverable ONLY here. Owner-only — no
-- persona grants; this table is the governable secret that makes
-- pseudonymization reversible. Survives init-db (additive, never dropped).
CREATE TABLE IF NOT EXISTS deid_vault (
    original_hash text PRIMARY KEY,
    entity_type   text NOT NULL,
    original      text NOT NULL,
    pseudonym     text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
-- Personas hold NO privilege on the disclosure log: run_query resets to the
-- owner before writing (D10; the dormant grants v1 issued were revoked in
-- migration 020). The audit row names the person by an opaque user id the
-- engine never reads (Phase 2 decision 3).
ALTER TABLE disclosure_log ADD COLUMN IF NOT EXISTS user_id bigint;
