-- Lane 2 staging: Synthea structured universe. Lives in its own schema —
-- this is the slice that migrates to Snowflake in the governance phase.
-- Applied by `raglab load-synthea` (drop-and-recreate, like schema.sql).

CREATE SCHEMA IF NOT EXISTS synthea;

DROP TABLE IF EXISTS synthea.claims;
DROP TABLE IF EXISTS synthea.medications;
DROP TABLE IF EXISTS synthea.conditions;
DROP TABLE IF EXISTS synthea.encounters;
DROP TABLE IF EXISTS synthea.enrollment;
DROP TABLE IF EXISTS synthea.patients;

CREATE TABLE synthea.patients (
    id         text PRIMARY KEY,
    birthdate  date,
    deathdate  date,
    ssn        text,
    first      text,
    last       text,
    marital    text,
    race       text,
    ethnicity  text,
    gender     text,
    address    text,
    city       text,
    state      text,
    zip        text,
    member_id  text UNIQUE,   -- natural key, derived (raglab identifiers)
    mrn        text UNIQUE
);

-- One row per member per plan year: the member's FEHB/PSHB enrollment.
-- Built deterministically by `raglab identifiers` (see raglab.enrollment).
CREATE TABLE synthea.enrollment (
    patient         text REFERENCES synthea.patients (id),
    member_id       text NOT NULL,
    year            integer NOT NULL,
    line_of_business text NOT NULL,          -- FEHB | PSHB
    plan_code       text NOT NULL,           -- 71-006 ...
    plan_option     text NOT NULL,           -- High | Standard | HDHP | Elevate | Elevate Plus
    tier            text NOT NULL,           -- Self Only | Self Plus One | Self and Family
    enrollment_code text NOT NULL,           -- FEHB enrollment code (e.g. 311)
    PRIMARY KEY (patient, year)
);

CREATE TABLE synthea.encounters (
    id                  text PRIMARY KEY,
    start               timestamptz,
    stop                timestamptz,
    patient             text REFERENCES synthea.patients (id),
    encounterclass      text,
    code                text,
    description         text,
    base_encounter_cost numeric,
    total_claim_cost    numeric,
    payer_coverage      numeric,
    reasoncode          text,
    reasondescription   text
);

CREATE TABLE synthea.conditions (
    start       date,
    stop        date,
    patient     text REFERENCES synthea.patients (id),
    encounter   text,
    code        text,
    description text
);

CREATE TABLE synthea.medications (
    start             timestamptz,
    stop              timestamptz,
    patient           text REFERENCES synthea.patients (id),
    encounter         text,
    code              text,
    description       text,
    base_cost         numeric,
    payer_coverage    numeric,
    dispenses         numeric,
    totalcost         numeric,
    reasoncode        text,
    reasondescription text
);

CREATE TABLE synthea.claims (
    id                 text PRIMARY KEY,
    patientid          text REFERENCES synthea.patients (id),
    providerid         text,
    currentillnessdate timestamptz,
    servicedate        timestamptz,
    diagnosis1         text,
    diagnosis2         text,
    healthcareclaimtypeid1 text
);

CREATE INDEX encounters_patient_idx  ON synthea.encounters (patient);
CREATE INDEX conditions_patient_idx  ON synthea.conditions (patient);
CREATE INDEX medications_patient_idx ON synthea.medications (patient);
CREATE INDEX claims_patient_idx      ON synthea.claims (patientid);
