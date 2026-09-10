"""Lane 2 entitlement contracts — the warehouse, not app code, shapes what
each role sees. Requires a live Snowflake account (skipped in CI, which
holds no secrets); run locally after `raglab snowflake-setup`.
"""

import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (Lane 2 verified locally, CI is zero-secret)",
)]


@pytest.fixture(scope="module")
def report():
    from raglab import snowlane

    sf = snowlane.connect()
    try:
        yield snowlane.verify(sf)
    finally:
        sf.close()


def test_examiner_sees_full_detail(report):
    r = report["CLAIMS_EXAMINER"]
    assert r["ssn_visible"] and r["cost_visible"]


def test_care_manager_financials_masked(report):
    r = report["CARE_MANAGER"]
    assert not r["cost_visible"], "care manager must not see financial columns"
    assert r["ssn_visible"], "clinical context keeps identity for care work"


def test_actuary_deidentified_but_aggregable(report):
    r = report["ACTUARY"]
    assert not r["ssn_visible"], "actuary must not see identifiers"
    assert not r["names_visible"] and r["sample_name"] is None, "D11: names are NULL for the actuary, not hashed"
    assert not r["city_visible"] and r["zip_length"] == 3, "D11: geography to minimum-necessary"
    assert r["distinct_patients"] == report["CLAIMS_EXAMINER"]["distinct_patients"], (
        "de-identification must not break member-level aggregates"
    )


def test_pshb_examiner_row_scoped(report):
    full, scoped = report["CLAIMS_EXAMINER"], report["PSHB_EXAMINER"]
    assert scoped["rows"] < full["rows"] and scoped["lobs"] == 1, (
        "row access policy must trim to one book of business"
    )
    assert scoped["ssn_visible"] and scoped["cost_visible"], (
        "row scoping composes with full column visibility"
    )


def test_operations_roles_see_amounts_but_not_ssn(report):
    """Phase 2 decision 4: reps and appeals analysts work from member IDs and
    amounts; neither verifies identity by SSN."""
    for role in ("MEMBER_SERVICES_REP", "APPEALS_ANALYST"):
        r = report[role]
        assert r["cost_visible"] and not r["ssn_visible"], role
        assert r["names_visible"] and r["lobs"] == 2, role
        assert r["distinct_patients"] == report["CLAIMS_EXAMINER"]["distinct_patients"], role
