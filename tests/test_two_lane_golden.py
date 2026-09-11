"""Compound golden questions executed end to end: each item's EXPECTED legs
run as a caller plan through planner.compose under the identity's Postgres
persona and Snowflake role — one payload, one disclosure, worst-of-required
status — and the required evidence must be present. (The planner's own
routing is scored in the eval as routing_accuracy; this test proves the
execution path.) Requires live Snowflake + the embedded corpus; skipped in
CI (zero-secret)."""

import json
import os

import psycopg
import pytest

from raglab import config, planner, snowlane
from raglab.ablation import load_golden
from raglab.mcp_server import IDENTITIES

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (verified locally, CI is zero-secret)",
)]

COMPOUND = [g for g in load_golden() if g["category"] == "compound"]


def _caller_plan(item: dict) -> planner.Plan:
    legs = []
    for i, leg in enumerate(item["expected_legs"]):
        if leg["kind"] == "doc_probe":
            legs.append({"name": f"docs{i}", "kind": "doc_probe", "text": leg.get("text") or item["question"], "sources": leg.get("sources", [])})
        else:
            q = leg.get("query_name") or leg["query_name_any"][0]
            spec = snowlane.NAMED_QUERIES[q]["params"]
            slots = [sname for sname in ("member_id", "claim_id", "case_id") if sname in spec]
            legs.append({"name": q, "kind": "member_query", "query_name": q, "slots": slots,
                         "params": (item.get("params") or {}).get(q, {})})
    return planner.plan_from_dict({"shape": "compound", "legs": legs})


@pytest.mark.parametrize("item", COMPOUND, ids=[g["id"] for g in COMPOUND])
def test_expected_legs_compose_under_identity(item):
    persona, role = IDENTITIES[item["identity"]]
    with psycopg.connect(config.DATABASE_URL) as conn:
        built = planner.compose(conn, item["question"], planner.Caller(persona=persona, warehouse_role=role),
                                member_id=item.get("member_id"), plan=_caller_plan(item), source="eval", module=item.get("module"))
    assert built["status"] == "ok", (built.get("missing"), [w.get("reason") for w in built["warehouse_results"]])
    titles = [c["source"]["title"] for c in built["chunks"]]
    need = item.get("required_evidence", {})
    for anchor in need.get("doc_anchors", []):
        assert any(anchor in t for t in titles), (anchor, titles[:8])
    doc_types = {c["source"].get("doc_type") for c in built["chunks"]}
    for src in need.get("doc_sources", []):
        assert src in doc_types, (src, doc_types)
    for q, n in need.get("rows_min", {}).items():  # "a|b": any listed query satisfies the minimum
        rows = [w for w in built["warehouse_results"] if w.get("query_name") in q.split("|") and w.get("status") == "ok"]
        assert rows and rows[0]["row_count"] >= n, (q, built["warehouse_results"])
    assert built["plan"]["legs"] and built["spec_version"] == "1.1.0"


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


