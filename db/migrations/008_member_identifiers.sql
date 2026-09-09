-- Stable member identifiers and enrollment on the Synthea population.
-- The synthea schema is created by `raglab load-synthea` (not by schema.sql),
-- so everything here is guarded: on a database without it (CI) this is a no-op.
DO $$
BEGIN
    IF to_regclass('synthea.patients') IS NOT NULL THEN
        ALTER TABLE synthea.patients ADD COLUMN IF NOT EXISTS member_id text UNIQUE;
        ALTER TABLE synthea.patients ADD COLUMN IF NOT EXISTS mrn text UNIQUE;
        CREATE TABLE IF NOT EXISTS synthea.enrollment (
            patient          text REFERENCES synthea.patients (id),
            member_id        text NOT NULL,
            year             integer NOT NULL,
            line_of_business text NOT NULL,
            plan_code        text NOT NULL,
            plan_option      text NOT NULL,
            tier             text NOT NULL,
            enrollment_code  text NOT NULL,
            PRIMARY KEY (patient, year)
        );
    END IF;
END
$$;
