"""Two-lane golden questions: each compound question decomposes into a
document probe (Lane 1, as the identity's Postgres persona) and a named
member-data query (Lane 2, as the identity's Snowflake role) — mirroring
how the MCP consumer actually works: the agent decomposes, each lane
enforces its own governance. Requires live Snowflake + the embedded corpus;
skipped in CI (zero-secret)."""

import json
import os

import psycopg
import pytest

from raglab import config, snowlane
from raglab.ablation import load_golden
from raglab.mcp_server import IDENTITIES
from raglab.pipeline import run_query

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (verified locally, CI is zero-secret)",
)]

TWO_LANE = [g for g in load_golden() if g["category"] == "two_lane"]


@pytest.mark.parametrize("item", TWO_LANE, ids=[g["id"] for g in TWO_LANE])
def test_both_lanes_answer_under_identity(item):
    lane1_persona, lane2_role = IDENTITIES[item["identity"]]

    with psycopg.connect(config.DATABASE_URL) as conn:
        built = run_query(conn, item["doc_probe"], persona=lane1_persona,
                          source="eval", member_id=item.get("member_id"))  # the surface supplies the member id
    assert built["status"] == "ok", built.get("confidence")
    titles = [c["source"]["title"] for c in built["chunks"][:5]]
    assert any(item["doc_anchor"] in t for t in titles), titles

    params = dict(item["member_query"])
    query_name = params.pop("query_name")
    sf = snowlane.connect(role=lane2_role)
    try:
        result = snowlane.run_named_query(sf, query_name, params)
    finally:
        sf.close()
    assert result["status"] == "ok" and result["row_count"] >= 1


def test_masking_composes_with_two_lane_answers():
    """The same member summary is answerable by care_team, but its financial
    columns are policy-masked — NULL is 'not visible to your role'."""
    sf = snowlane.connect(role="CARE_MANAGER")
    try:
        result = snowlane.run_named_query(
            sf, "member_claims_summary",
            {"last_name": "Dickinson", "first_name": "Karima"},
        )
    finally:
        sf.close()
    row = dict(zip(result["columns"], result["rows"][0]))
    assert row["CLAIM_LINES"] >= 1
    assert row["TOTAL_COST"] is None and row["PAYER_COVERAGE"] is None
    assert set(result["masked_columns"]) == {"TOTAL_COST", "PAYER_COVERAGE"}, (
        "the response must distinguish policy-masked columns from data NULLs"
    )


def test_actuary_aggregate_is_deidentified():
    sf = snowlane.connect(role="ACTUARY")
    try:
        result = snowlane.run_named_query(
            sf, "cost_by_condition", {"description_like": "%asthma%"}
        )
    finally:
        sf.close()
    assert result["row_count"] >= 1
    assert "MEMBERS" in result["columns"]  # distinct-member counts survive
    assert not any("NAME" in c for c in result["columns"])


