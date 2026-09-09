"""Load Synthea CSV exports into the synthea schema via COPY.

Synthea's CSV column sets vary across releases, so each table loads the
intersection of the CSV header and our declared columns — extra CSV columns
are ignored, missing ones load as NULL."""

import csv
from dataclasses import dataclass
from pathlib import Path

import psycopg

from raglab import config

CSV_DIR = config.REPO_ROOT / "data" / "member" / "synthea" / "csv"
SYNTHEA_SCHEMA_PATH = config.REPO_ROOT / "db" / "synthea.sql"

# table -> (csv filename, declared columns in table order)
TABLES = {
    "patients": (
        "patients.csv",
        ["id", "birthdate", "deathdate", "ssn", "first", "last", "marital",
         "race", "ethnicity", "gender", "address", "city", "state", "zip"],
    ),
    "encounters": (
        "encounters.csv",
        ["id", "start", "stop", "patient", "encounterclass", "code",
         "description", "base_encounter_cost", "total_claim_cost",
         "payer_coverage", "reasoncode", "reasondescription",
         "organization", "provider"],
    ),
    "organizations": (
        "organizations.csv",
        ["id", "name", "address", "city", "state", "zip", "phone", "npi"],
    ),
    "providers": (
        "providers.csv",
        ["id", "organization", "name", "gender", "speciality", "address", "city", "state", "zip", "npi"],
    ),
    "conditions": (
        "conditions.csv",
        ["start", "stop", "patient", "encounter", "code", "description"],
    ),
    "medications": (
        "medications.csv",
        ["start", "stop", "patient", "encounter", "code", "description",
         "base_cost", "payer_coverage", "dispenses", "totalcost",
         "reasoncode", "reasondescription"],
    ),
    "claims": (
        "claims.csv",
        ["id", "patientid", "providerid", "currentillnessdate", "servicedate",
         "diagnosis1", "diagnosis2", "healthcareclaimtypeid1"],
    ),
}


@dataclass
class TableLoad:
    table: str
    rows: int
    skipped_csv_columns: int


def load_all(conn: psycopg.Connection, csv_dir: Path = CSV_DIR) -> list[TableLoad]:
    conn.execute(SYNTHEA_SCHEMA_PATH.read_text())
    results = []
    for table, (filename, declared) in TABLES.items():
        path = csv_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing Synthea export: {path}")
        results.append(_load_table(conn, table, path, declared))
    conn.commit()
    return results


def _load_table(
    conn: psycopg.Connection, table: str, path: Path, declared: list[str]
) -> TableLoad:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        use = [c for c in declared if c in header]
        positions = [header.index(c) for c in use]
        columns = ", ".join(f'"{c}"' for c in use)
        rows = 0
        with conn.cursor() as cur:
            with cur.copy(
                f"COPY synthea.{table} ({columns}) FROM STDIN"
            ) as copy:
                for row in reader:
                    copy.write_row(
                        [row[p] if row[p] != "" else None for p in positions]
                    )
                    rows += 1
    return TableLoad(table=table, rows=rows, skipped_csv_columns=len(header) - len(use))
