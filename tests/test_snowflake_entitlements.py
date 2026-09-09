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
    assert r["sample_name"] and len(r["sample_name"]) == 64, (
        "names must be stable hashes so distinct-member aggregation works"
    )
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
