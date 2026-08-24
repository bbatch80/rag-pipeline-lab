-- Lane 2 staging: Synthea structured universe. Lives in its own schema —
-- this is the slice that migrates to Snowflake in the governance phase.
-- Applied by `raglab load-synthea` (drop-and-recreate, like schema.sql).

CREATE SCHEMA IF NOT EXISTS synthea;

DROP TABLE IF EXISTS synthea.claims;
DROP TABLE IF EXISTS synthea.medications;
DROP TABLE IF EXISTS synthea.conditions;
DROP TABLE IF EXISTS synthea.encounters;
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
    zip        text
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
