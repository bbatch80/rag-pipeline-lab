-- Governance layer: lateral need-to-know RLS + disclosure log.
-- Applied by `raglab init-db` after schema.sql (policies attach to the
-- recreated tables; roles are cluster-level and persist).
--
-- Access model (HIPAA minimum-necessary):
--   public    -> acl_tag = 'public' only
--   employee  -> public + employee (SOPs, bulletins, formulary, KB)
--   care_team -> public + care_team (clinical notes)
-- Neither non-public tier sees the other. The engine filters BEFORE ranking:
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
END
$$;

-- The app user may assume any persona (SET ROLE); with no persona set it is
-- a member of all three, i.e. the administrative full view.
GRANT persona_public, persona_employee, persona_care_team TO raglab;

GRANT SELECT ON documents, chunks, lexeme_df TO
    persona_public, persona_employee, persona_care_team;

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS chunks_lateral_acl ON chunks;
CREATE POLICY chunks_lateral_acl ON chunks FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'
        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'
        AND pg_has_role(current_user, 'persona_care_team', 'member'))
);

DROP POLICY IF EXISTS documents_lateral_acl ON documents;
CREATE POLICY documents_lateral_acl ON documents FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'
        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'
        AND pg_has_role(current_user, 'persona_care_team', 'member'))
);

-- Writes stay owner-only: personas are read-only consumers.

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
GRANT SELECT, INSERT ON disclosure_log TO
    persona_public, persona_employee, persona_care_team;
GRANT USAGE ON SEQUENCE disclosure_log_id_seq TO
    persona_public, persona_employee, persona_care_team;
