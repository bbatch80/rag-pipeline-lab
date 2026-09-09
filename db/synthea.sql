-- Lane 2 staging: Synthea structured universe. Lives in its own schema —
-- this is the slice that migrates to Snowflake in the governance phase.
-- Applied by `raglab load-synthea` (drop-and-recreate, like schema.sql).

CREATE SCHEMA IF NOT EXISTS synthea;

DROP TABLE IF EXISTS synthea.appeals;
DROP TABLE IF EXISTS synthea.claim_adjudication;
DROP TABLE IF EXISTS synthea.claims;
DROP TABLE IF EXISTS synthea.medications;
DROP TABLE IF EXISTS synthea.conditions;
DROP TABLE IF EXISTS synthea.encounters;
DROP TABLE IF EXISTS synthea.call_log;
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

-- Call-center interaction records (fields; the note text is a vector-lane
-- document). Written by `raglab synth calls`.
CREATE TABLE synthea.call_log (
    call_id      text PRIMARY KEY,
    patient      text REFERENCES synthea.patients (id),
    member_id    text NOT NULL,
    call_date    date NOT NULL,
    rep_id       text NOT NULL,
    reason_code  text NOT NULL,
    disposition  text NOT NULL,
    claim_id     text,
    duration_sec integer NOT NULL
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

-- Claim adjudication overlay + appeals (P1-PR4; mirrors migration 013).
CREATE TABLE synthea.claim_adjudication (
    encounter     text PRIMARY KEY REFERENCES synthea.encounters (id),
    claim_id      text NOT NULL,
    patient       text NOT NULL REFERENCES synthea.patients (id),
    member_id     text NOT NULL,
    status        text NOT NULL CHECK (status IN ('paid', 'denied', 'pending')),
    decision_date date,
    denial_reason text,
    policy_id     text
);
CREATE TABLE synthea.appeals (
    case_id       text PRIMARY KEY,
    patient       text NOT NULL REFERENCES synthea.patients (id),
    member_id     text NOT NULL,
    encounter     text NOT NULL REFERENCES synthea.claim_adjudication (encounter),
    claim_id      text NOT NULL,
    call_id       text REFERENCES synthea.call_log (call_id),
    filed_date    date NOT NULL,
    appeal_type   text NOT NULL,
    denial_reason text NOT NULL,
    decision      text NOT NULL,
    decided_date  date,
    reviewer      text,
    policy_id     text
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
