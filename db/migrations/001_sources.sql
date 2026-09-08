-- The fixed inventory of data sources. Eighteen rows, seeded here, never
-- added to by the application: there is no registry or intake mechanism.
-- Columns replace the per-doc-type literals that used to live in code
-- (gate rules, PHI flag, churn pool, directory layout).
CREATE TABLE IF NOT EXISTS sources (
    source_id           smallint PRIMARY KEY,
    key                 text NOT NULL UNIQUE,
    display_name        text NOT NULL,
    department          text NOT NULL,
    lane                text NOT NULL CHECK (lane IN ('vector', 'warehouse', 'both')),
    acl_tag             text,                       -- vector lane only
    doc_type            text UNIQUE,                -- chunks.doc_type value; vector lane only
    dir                 text,                       -- repo-relative directory of source files
    parser              text,                       -- pdf | markdown | csv | generator
    chunk_profile       text CHECK (chunk_profile IN ('section', 'record', 'row')),
    gate_min_chunks     integer NOT NULL DEFAULT 1,
    gate_median_low     integer,
    gate_median_high    integer,
    gate_expected_terms text[] NOT NULL DEFAULT '{}',
    phi                 boolean NOT NULL DEFAULT false,
    cadence             text NOT NULL CHECK (cadence IN ('static', 'annual', 'daily', 'continuous')),
    churn_eligible      boolean NOT NULL DEFAULT false,
    status              text NOT NULL CHECK (status IN ('ingested', 'loaded', 'planned'))
);

INSERT INTO sources
    (source_id, key, display_name, department, lane, acl_tag, doc_type, dir, parser,
     chunk_profile, gate_min_chunks, gate_median_low, gate_median_high,
     gate_expected_terms, phi, cadence, churn_eligible, status)
VALUES
    (1,  'brochures',         'FEHB/PSHB plan brochures',   'Plan Documents',   'vector',    'public',          'brochure',        'data/raw',                   'pdf',       'section', 50, 250, 1600, '{out-of-pocket,deductible}', false, 'annual',     false, 'ingested'),
    (2,  'rates',             'Premium rates',              'Plan Documents',   'vector',    'public',          'rates',           'data/internal/rates',        'csv',       'row',      1, NULL, NULL, '{}', false, 'annual',     false, 'ingested'),
    (3,  'sops',              'Standard operating procedures', 'Member Services', 'vector',  'employee',        'sop',             'data/internal/sops',         'markdown',  'section',  1,  60, 1900, '{}', false, 'static',     false, 'ingested'),
    (4,  'bulletins',         'Claims bulletins',           'Claims',           'vector',    'employee',        'bulletin',        'data/internal/bulletins',    'markdown',  'section',  1,  60, 1900, '{}', false, 'static',     false, 'ingested'),
    (5,  'formulary',         'Formulary',                  'Pharmacy',         'vector',    'employee',        'formulary',       'data/internal/formulary',    'markdown',  'section',  1, NULL, NULL, '{}', false, 'static',     true,  'ingested'),
    (6,  'kb',                'CSR knowledge base',         'Member Services',  'vector',    'employee',        'kb',              'data/internal/kb',           'markdown',  'section',  1, NULL, NULL, '{}', false, 'static',     true,  'ingested'),
    (7,  'clinical_notes',    'Clinical notes',             'Care Management',  'vector',    'care_team',       'clinical_note',   'data/internal/notes',        'markdown',  'section',  1,  60, 1900, '{}', true,  'static',     false, 'ingested'),
    (8,  'call_notes',        'Call notes',                 'Member Services',  'vector',    'member_services', 'call_note',       'data/internal/calls',        'markdown',  'record',   1, NULL, NULL, '{}', true,  'continuous', false, 'planned'),
    (9,  'appeal_documents',  'Appeals case documents',     'Appeals',          'vector',    'appeals',         'appeal',          'data/internal/appeals',      'markdown',  'section',  1, NULL, NULL, '{}', true,  'daily',      false, 'planned'),
    (10, 'clinical_policies', 'Clinical policies',          'Medical Policy',   'vector',    'public',          'clinical_policy', 'data/internal/policies',     'markdown',  'section',  1, NULL, NULL, '{}', false, 'static',     false, 'planned'),
    (11, 'carrier_letters',   'OPM carrier letters',        'Plan Documents',   'vector',    'public',          'carrier_letter',  'data/raw/carrier_letters',   'pdf',       'section',  1, NULL, NULL, '{}', false, 'annual',     false, 'planned'),
    (12, 'patients',          'PATIENTS',                   'Enrollment',       'warehouse', NULL, NULL, NULL, 'generator', NULL, 1, NULL, NULL, '{}', true,  'static',     false, 'loaded'),
    (13, 'claim_lines',       'CLAIM_LINES',                'Claims',           'warehouse', NULL, NULL, NULL, 'generator', NULL, 1, NULL, NULL, '{}', true,  'daily',      false, 'loaded'),
    (14, 'call_log',          'CALL_LOG',                   'Member Services',  'warehouse', NULL, NULL, NULL, 'generator', NULL, 1, NULL, NULL, '{}', true,  'continuous', false, 'planned'),
    (15, 'appeals',           'APPEALS',                    'Appeals',          'warehouse', NULL, NULL, NULL, 'generator', NULL, 1, NULL, NULL, '{}', true,  'daily',      false, 'planned'),
    (16, 'claim_adjudication','CLAIM_ADJUDICATION',         'Claims',           'warehouse', NULL, NULL, NULL, 'generator', NULL, 1, NULL, NULL, '{}', false, 'daily',      false, 'planned'),
    (17, 'providers',         'PROVIDERS + ORGANIZATIONS',  'Provider Network', 'warehouse', NULL, NULL, NULL, 'csv',       NULL, 1, NULL, NULL, '{}', false, 'static',     false, 'planned'),
    (18, 'enrollment',        'ENROLLMENT',                 'Enrollment',       'warehouse', NULL, NULL, NULL, 'generator', NULL, 1, NULL, NULL, '{}', true,  'annual',     false, 'planned');

DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'persona_public') THEN
        GRANT SELECT ON sources TO persona_public, persona_employee, persona_care_team;
    END IF;
END
$$;
