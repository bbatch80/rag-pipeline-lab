-- Housekeeping (Phase 1 exit check): the enrollment table has been loaded
-- since P1-PR2 (raglab identifiers); the registry row never recorded it.
UPDATE sources SET status = 'loaded' WHERE key = 'enrollment';
