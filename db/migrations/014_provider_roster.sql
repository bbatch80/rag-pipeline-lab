-- Provider roster (P1-PR5). Guarded — no synthea schema in CI.
--
-- Synthea's providers.csv and organizations.csv: the payer's provider
-- directory, minus contract detail. Network participation is synthesized at
-- the ORGANIZATION × plan level (contracts are signed by organizations, not
-- clinicians) and inherited by every provider of the organization — about
-- 90% in-network per plan, deterministic from a hash of organization id +
-- plan code (raglab.roster). Stated simplifications: no contract dates, no
-- directory-verification date.
--
-- Encounters gain the provider and organization Synthea records on every
-- encounter (dropped in v1 because nothing used them): claim lines then
-- know their provider, which is what an out-of-network denial or a rep's
-- "is this provider participating?" needs. Backfilled in place by encounter
-- id (adjudication and appeals hold foreign keys to encounters).
DO $$
BEGIN
    IF to_regclass('synthea.patients') IS NOT NULL THEN
        CREATE TABLE IF NOT EXISTS synthea.organizations (
            id      text PRIMARY KEY,
            name    text NOT NULL,
            address text,
            city    text,
            state   text,
            zip     text,
            phone   text,
            npi     text
        );
        CREATE TABLE IF NOT EXISTS synthea.providers (
            id           text PRIMARY KEY,
            organization text REFERENCES synthea.organizations (id),
            name         text NOT NULL,
            gender       text,
            speciality   text,
            address      text,
            city         text,
            state        text,
            zip          text,
            npi          text
        );
        CREATE INDEX IF NOT EXISTS providers_npi_idx ON synthea.providers (npi);
        CREATE INDEX IF NOT EXISTS providers_speciality_idx ON synthea.providers (speciality);
        CREATE TABLE IF NOT EXISTS synthea.provider_network (
            organization text NOT NULL REFERENCES synthea.organizations (id),
            plan_code    text NOT NULL,
            in_network   boolean NOT NULL,
            PRIMARY KEY (organization, plan_code)
        );
        ALTER TABLE synthea.encounters ADD COLUMN IF NOT EXISTS provider text;
        ALTER TABLE synthea.encounters ADD COLUMN IF NOT EXISTS organization text;
    END IF;
END
$$;
UPDATE sources SET status = 'loaded' WHERE key = 'providers';
