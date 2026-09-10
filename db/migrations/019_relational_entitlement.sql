-- Relational entitlement (P2-PR2, Phase 2 decision 2): a clinical note is
-- visible to the appeals tier when ANY appeal cites it in appeal_evidence,
-- and disappears when the citation is removed — the first entitlement that
-- depends on data state, not role membership. Zero re-indexing: the row
-- policies read the link table at query time. Keyed by document TITLE
-- (titles survive a re-ingest; document ids do not); the EXISTS is not
-- pushed below RLS, so documents(title) and appeal_evidence(document_title)
-- are indexed. Personas can read the link table (it holds case ids and
-- titles, no clinical content); the appeals tier can never browse the
-- clinical corpus.
CREATE INDEX IF NOT EXISTS documents_title_idx ON documents (title);
GRANT SELECT ON appeal_evidence TO persona_public, persona_employee, persona_care_team, persona_member_services, persona_appeals;

DROP POLICY IF EXISTS documents_lateral_acl ON documents;
CREATE POLICY documents_lateral_acl ON documents FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'         AND pg_has_role(current_user, 'persona_appeals', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_appeals', 'member')
        AND EXISTS (SELECT 1 FROM appeal_evidence e
                    WHERE e.document_title = documents.title AND e.kind = 'clinical_note'))
);
DROP POLICY IF EXISTS chunks_lateral_acl ON chunks;
CREATE POLICY chunks_lateral_acl ON chunks FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'         AND pg_has_role(current_user, 'persona_appeals', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_appeals', 'member')
        AND EXISTS (SELECT 1 FROM appeal_evidence e JOIN documents d ON d.title = e.document_title
                    WHERE d.id = chunks.document_id AND e.kind = 'clinical_note'))
);
