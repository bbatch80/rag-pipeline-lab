-- A bulletin number names one issuance, so it is the leaf of its hierarchy:
-- a question that types "Bulletin 2026-002" is asking about that document,
-- superseded or not, and the version-precedence window does not hide it.
-- (A policy id names a FAMILY of versions — policy_id has 'version' beneath
-- it — so the window still applies there.)
UPDATE sources SET hierarchy = '{bulletin_id}' WHERE key = 'bulletins';
