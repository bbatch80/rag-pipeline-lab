-- First start of a deployed database: the cluster-level roles the snapshot's
-- policies and grants reference (pg_dump carries neither roles nor their
-- memberships). Mirrors db/governance.sql; the snapshot restores the rest.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'persona_public') THEN CREATE ROLE persona_public NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'persona_employee') THEN CREATE ROLE persona_employee NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'persona_care_team') THEN CREATE ROLE persona_care_team NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'persona_member_services') THEN CREATE ROLE persona_member_services NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'persona_appeals') THEN CREATE ROLE persona_appeals NOLOGIN; END IF;
END $$;
GRANT persona_employee TO persona_member_services;
GRANT persona_employee TO persona_appeals;
