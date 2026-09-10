-- Relational entitlement, second record type: a call note is visible to the
-- appeals tier when ANY appeal cites it in appeal_evidence (kind='call_note'),
-- exactly as migration 019 did for clinical notes. Same key (document title),
-- same link table, no re-indexing. The appeals tier still cannot browse the
-- call-note corpus; it reads only the calls that are part of a case file.
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
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_appeals', 'member')
        AND EXISTS (SELECT 1 FROM appeal_evidence e
                    WHERE e.document_title = documents.title AND e.kind = 'call_note'))
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
    OR (acl_tag = 'member_services' AND pg_has_role(current_user, 'persona_appeals', 'member')
        AND EXISTS (SELECT 1 FROM appeal_evidence e JOIN documents d ON d.title = e.document_title
                    WHERE d.id = chunks.document_id AND e.kind = 'call_note'))
);
