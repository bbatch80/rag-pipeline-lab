-- Two facts about a source that retrieval needs:
--   member_scoped — its documents are records about one person. They are
--     searched only inside a member context (the member the caller has
--     open, or a member identifier in the question) and filtered to that
--     member; with no member context they are not searched at all.
--   event — its documents are dated events (a call), not yearly editions
--     (a brochure). The router's year filter is an edition filter and does
--     not apply to them: a question about a 2025 claim may be answered by a
--     call placed in 2026.
ALTER TABLE sources ADD COLUMN IF NOT EXISTS member_scoped boolean NOT NULL DEFAULT false;
ALTER TABLE sources ADD COLUMN IF NOT EXISTS event boolean NOT NULL DEFAULT false;
UPDATE sources SET member_scoped = true, event = true WHERE key = 'call_notes';

-- Call notes carry the year of the call, not the year they were ingested.
UPDATE chunks c SET year = EXTRACT(YEAR FROM l.call_date)::int
FROM documents d, synthea.call_log l
WHERE c.document_id = d.id AND c.doc_type = 'call_note'
  AND l.call_id = replace(d.title, 'call_', '')
  AND c.year IS DISTINCT FROM EXTRACT(YEAR FROM l.call_date)::int;
