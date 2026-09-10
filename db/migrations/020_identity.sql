-- Identity model (P2-PR3, Phase 2 decision 3). Resolved at the EDGE (the
-- MCP server now, the login session in Phase 4): a username becomes a
-- document persona, a warehouse role, and an opaque user id. The engine
-- receives those three and stamps the id into the disclosure row; the row
-- policies stay role-only and never learn a user. Groups carry the
-- entitlement (persona + warehouse role) and the Phase 5 surface grants.
CREATE TABLE IF NOT EXISTS groups (
    name           text PRIMARY KEY,
    persona        text NOT NULL,     -- document-lane persona (pipeline.PERSONAS)
    warehouse_role text,              -- Snowflake role; NULL = no member-data access
    description    text
);
CREATE TABLE IF NOT EXISTS group_surfaces (
    group_name text NOT NULL REFERENCES groups (name) ON DELETE CASCADE,
    surface    text NOT NULL,         -- ask | agent_assist | appeals_workbench | analyst_view | console
    PRIMARY KEY (group_name, surface)
);
CREATE TABLE IF NOT EXISTS users (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username      text UNIQUE NOT NULL,
    display_name  text NOT NULL,
    password_hash text NOT NULL,      -- scrypt (stdlib), prototype-grade login (Phase 4)
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS user_groups (
    user_id    bigint NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    group_name text NOT NULL REFERENCES groups (name) ON DELETE CASCADE,
    PRIMARY KEY (user_id, group_name)
);

-- The audit row names the person (opaque id; the engine never reads it).
ALTER TABLE disclosure_log ADD COLUMN IF NOT EXISTS user_id bigint;

-- D10: the persona roles never exercised their disclosure_log grants
-- (run_query resets to the owner before writing); a dormant privilege is
-- how a future persona-scoped query would silently work. Revoke.
REVOKE ALL ON disclosure_log FROM persona_public, persona_employee, persona_care_team, persona_member_services, persona_appeals;
REVOKE ALL ON SEQUENCE disclosure_log_id_seq FROM persona_public, persona_employee, persona_care_team, persona_member_services, persona_appeals;
