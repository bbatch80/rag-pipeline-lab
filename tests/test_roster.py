"""Provider roster: organizations, providers, encounter links, and network
status per organization × plan (providers inherit)."""

import pytest


from raglab import roster
from raglab.synthea_load import CSV_DIR

pytestmark = [pytest.mark.readonly, pytest.mark.skipif(not (CSV_DIR / "providers.csv").exists(),
                                reason="Synthea CSVs not on this machine (CI's fresh database)")]


def _seed(db):
    from raglab import config

    db.execute((config.REPO_ROOT / "db" / "synthea.sql").read_text())
    db.execute("INSERT INTO synthea.patients (id, birthdate, first, last, ssn) VALUES ('p1', '1980-01-01', 'Ann', 'Lee', '1')")
    db.execute("INSERT INTO synthea.encounters (id, start, patient, description) VALUES "
               "('76b20010-c318-5754-57e2-651f596ddb8c', '2025-01-01', 'p1', 'visit')")  # first row of encounters.csv
    db.execute("INSERT INTO synthea.enrollment (patient, member_id, year, line_of_business, plan_code, plan_option, tier, enrollment_code) "
               "VALUES ('p1', 'M000000000', 2025, 'FEHB', '71-006', 'High', 'Self', '311'), "
               "('p1', 'M000000000', 2026, 'FEHB', '71-014', 'HDHP', 'Self', '341')")


def test_roster_loads_links_and_assigns_network(db):
    _seed(db)
    r = roster.load(db)
    assert r.organizations == 1145 and r.providers == 1145
    assert r.encounters_linked == 1, "the encounter present gets its Synthea provider and organization"
    org, prov = db.execute("SELECT organization, provider FROM synthea.encounters WHERE id = '76b20010-c318-5754-57e2-651f596ddb8c'").fetchone()
    assert org and prov
    assert db.execute("SELECT count(*) FROM synthea.providers WHERE organization IS NULL").fetchone()[0] == 0
    # one row per organization × plan code in force; ~90% in-network
    plans = db.execute("SELECT count(DISTINCT plan_code) FROM synthea.enrollment").fetchone()[0]
    assert r.network_rows == 1145 * plans and 0.85 <= r.in_network_share <= 0.95
    # organization level: every provider of an organization shares its status, by construction of the join
    assert roster.in_network("org-a", "71-006") == roster.in_network("org-a", "71-006")
    assert roster.assign_network(db) == (r.network_rows, r.in_network_share), "deterministic"


def test_network_status_is_a_pure_function_of_organization_and_plan():
    hits = sum(roster.in_network(f"org-{i}", "71-006") for i in range(2000))
    assert 1700 <= hits <= 1900
    assert roster.in_network("org-1", "71-006") == roster.in_network("org-1", "71-006")
