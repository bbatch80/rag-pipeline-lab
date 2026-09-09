"""Provider roster: Synthea providers + organizations into the synthea
schema, the encounter → provider/organization links Synthea carries on every
encounter, and synthesized network participation.

Network status is decided per ORGANIZATION × plan code — contracts are
signed by organizations, not clinicians — and inherited by every provider
of the organization. About 90% in-network per plan, deterministic from a
hash of organization id + plan code, so loading more data never flips a
status. No text, no member PHI (provider names stay with the statistical
recognizer)."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg

from raglab import identifiers
from raglab.synthea_load import CSV_DIR

IN_NETWORK_RATE = 0.9

ORGANIZATION_COLUMNS = ["id", "name", "address", "city", "state", "zip", "phone", "npi"]
PROVIDER_COLUMNS = ["id", "organization", "name", "gender", "speciality", "address", "city", "state", "zip", "npi"]


@dataclass
class RosterLoad:
    organizations: int
    providers: int
    encounters_linked: int
    network_rows: int
    in_network_share: float


def _rows(path: Path, columns: list[str]) -> list[list]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        positions = [header.index(c) for c in columns]
        return [[row[p] if row[p] != "" else None for p in positions] for row in reader]


def load(conn: psycopg.Connection, csv_dir: Path = CSV_DIR) -> RosterLoad:
    """Organizations, providers, the encounter links (backfilled in place by
    encounter id), and the network table."""
    conn.execute("DELETE FROM synthea.provider_network")
    conn.execute("DELETE FROM synthea.providers")
    conn.execute("DELETE FROM synthea.organizations")
    orgs = _rows(csv_dir / "organizations.csv", ORGANIZATION_COLUMNS)
    provs = _rows(csv_dir / "providers.csv", PROVIDER_COLUMNS)
    with conn.cursor() as cur:
        with cur.copy(f"COPY synthea.organizations ({', '.join(ORGANIZATION_COLUMNS)}) FROM STDIN") as copy:
            for row in orgs:
                copy.write_row(row)
        with cur.copy(f"COPY synthea.providers ({', '.join(PROVIDER_COLUMNS)}) FROM STDIN") as copy:
            for row in provs:
                copy.write_row(row)
    linked = link_encounters(conn, csv_dir)
    network, share = assign_network(conn)
    return RosterLoad(len(orgs), len(provs), linked, network, share)


def link_encounters(conn: psycopg.Connection, csv_dir: Path = CSV_DIR) -> int:
    """encounters.provider / .organization from encounters.csv, by id — an
    UPDATE, not a reload: adjudication and appeals reference encounters."""
    conn.execute("CREATE TEMP TABLE encounter_links (id text PRIMARY KEY, organization text, provider text) ON COMMIT DROP")
    with open(csv_dir / "encounters.csv", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        pid, porg, pprov = header.index("id"), header.index("organization"), header.index("provider")
        with conn.cursor() as cur:
            with cur.copy("COPY encounter_links (id, organization, provider) FROM STDIN") as copy:
                for row in reader:
                    copy.write_row([row[pid], row[porg] or None, row[pprov] or None])
    result = conn.execute(
        "UPDATE synthea.encounters e SET provider = l.provider, organization = l.organization "
        "FROM encounter_links l WHERE l.id = e.id "
        "AND (e.provider IS DISTINCT FROM l.provider OR e.organization IS DISTINCT FROM l.organization)"
    )
    return result.rowcount


def in_network(organization: str, plan_code: str) -> bool:
    """Deterministic participation for an organization under a plan."""
    h = int(hashlib.sha256(f"{identifiers.SEED}|network|{organization}|{plan_code}".encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000 < IN_NETWORK_RATE


def assign_network(conn: psycopg.Connection) -> tuple[int, float]:
    """One row per organization × plan code in force (the enrollment table's
    plan codes); providers inherit their organization's status."""
    plans = [r[0] for r in conn.execute("SELECT DISTINCT plan_code FROM synthea.enrollment ORDER BY 1").fetchall()]
    orgs = [r[0] for r in conn.execute("SELECT id FROM synthea.organizations ORDER BY id").fetchall()]
    rows = [(org, plan, in_network(org, plan)) for org in orgs for plan in plans]
    conn.execute("DELETE FROM synthea.provider_network")
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO synthea.provider_network (organization, plan_code, in_network) VALUES (%s, %s, %s)", rows
        )
    share = sum(1 for _, _, x in rows if x) / len(rows) if rows else 0.0
    return len(rows), round(share, 3)
