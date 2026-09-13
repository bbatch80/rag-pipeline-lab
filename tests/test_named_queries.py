"""Warehouse roles × masking policies, run live: the same catalog query
returns the same rows to every role while the role's policy nulls the
columns it may not see — costs for the care manager, identifiers for the
actuary, the SSN for every operations role. Requires live Snowflake; the
golden named_query items exercise the same queries through the composed
path in the eval."""

import os

import pytest

from raglab import snowlane
from raglab.snowlane import MASKED_FOR_ROLE

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (verified locally, CI is zero-secret)",
)]

MEMBER = "M767394984"  # golden named_query-05's member: 35 claim lines


@pytest.mark.parametrize("role", ["CLAIMS_EXAMINER", "CARE_MANAGER", "ACTUARY"])
def test_claims_summary_masks_by_role(role):
    sf = snowlane.connect(role=role)
    try:
        result = snowlane.run_named_query(sf, "member_claims_summary", {"member_id": MEMBER})
    finally:
        sf.close()
    assert result["status"] == "ok", result
    row = dict(zip(result["columns"], result["rows"][0])) if result["rows"] else {}
    expected_masked = MASKED_FOR_ROLE.get(role, set()) & set(result["columns"])
    assert set(result["masked_columns"]) == expected_masked, (role, result["masked_columns"])
    for col in expected_masked:
        assert row.get(col) is None, (role, col, "masked column carried a value")
    if role == "CLAIMS_EXAMINER":
        assert row["CLAIM_LINES"] == 35 and row["TOTAL_COST"] is not None
    if role == "ACTUARY":
        # The mask nulls MEMBER_ID before the predicate runs, so a lookup by id
        # finds nothing: an actuary cannot profile one member (status stays ok,
        # zero rows — the silent empty-ok is logged as an open failure mode).
        assert result["row_count"] == 0 and result["rows"] == [], "the identifier mask hides the individual"
