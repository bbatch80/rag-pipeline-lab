-- Phase 2 tiers (P2-PR1): member_services (call notes) and appeals (appeal
-- documents) replace the provisional employee tag. Tiers compose by role
-- inheritance: both new roles are members of persona_employee, so one
-- SET ROLE grants public + employee + the role's own tier through the
-- existing pg_has_role checks (decision 1: the rep's material — CSR KB,
-- SOPs — is employee-tier). Re-tagging is a metadata change on a typed
-- column: no re-ingest. Roles are cluster-level and idempotent.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_member_services') THEN
        CREATE ROLE persona_member_services NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_appeals') THEN
        CREATE ROLE persona_appeals NOLOGIN;
    END IF;
END
$$;
GRANT persona_employee TO persona_member_services, persona_appeals;
GRANT persona_member_services, persona_appeals TO raglab;
GRANT SELECT ON documents, chunks TO persona_member_services, persona_appeals;

DROP POLICY IF EXISTS chunks_lateral_acl ON chunks;
CREATE POLICY chunks_lateral_acl ON chunks FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'         AND pg_has_role(current_user, 'persona_appeals', 'member'))
);
DROP POLICY IF EXISTS documents_lateral_acl ON documents;
CREATE POLICY documents_lateral_acl ON documents FOR SELECT USING (
    acl_tag = 'public'
    OR (acl_tag = 'employee'        AND pg_has_role(current_user, 'persona_employee', 'member'))
    OR (acl_tag = 'care_team'       AND pg_has_role(current_user, 'persona_care_team', 'member'))
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_member_services', 'member'))
    OR (acl_tag = 'appeals'         AND pg_has_role(current_user, 'persona_appeals', 'member'))
);

UPDATE sources SET acl_tag = 'member_services' WHERE key = 'call_notes';
UPDATE sources SET acl_tag = 'appeals' WHERE key = 'appeal_documents';
UPDATE documents d SET acl_tag = s.acl_tag FROM sources s
WHERE s.source_id = d.source_id AND s.key IN ('call_notes', 'appeal_documents') AND d.acl_tag IS DISTINCT FROM s.acl_tag;
UPDATE chunks c SET acl_tag = d.acl_tag FROM documents d
WHERE d.id = c.document_id AND c.doc_type IN ('call_note', 'appeal') AND c.acl_tag IS DISTINCT FROM d.acl_tag;
