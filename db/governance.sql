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
    -- relational (Phase 2 decision 2): a clinical note any appeal cites
    OR (acl_tag = 'care_team'
        AND pg_has_role(current_user, 'persona_appeals', 'member')
        AND EXISTS (SELECT 1 FROM appeal_evidence e JOIN documents d ON d.title = e.document_title
                    WHERE d.id = chunks.document_id AND e.kind = 'clinical_note'))
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
    -- relational (Phase 2 decision 2): a clinical note any appeal cites
    OR (acl_tag = 'care_team'
        AND pg_has_role(current_user, 'persona_appeals', 'member')
        AND EXISTS (SELECT 1 FROM appeal_evidence e
                    WHERE e.document_title = documents.title AND e.kind = 'clinical_note'))
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
