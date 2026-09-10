-- Clinical notes are records about one person: a member question must never
-- return another patient's note. The source was registered without the
-- member_scoped flag (found by golden item S1, Phase 2 governance slice);
-- flipping it applies the existing member filter — chunks already carry
-- member_key — with no re-indexing.
UPDATE sources SET member_scoped = true WHERE key = 'clinical_notes';
