"""Named-query golden items (category named_query): warehouse-only
assertions — the roster and enrollment lookups a rep composes — run under
the identity's Snowflake role. Each item lists its queries in order and,
per query, what the result must contain. Requires live Snowflake."""

import os

import pytest

from raglab import snowlane
from raglab.ablation import load_golden
from raglab.mcp_server import IDENTITIES

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (verified locally, CI is zero-secret)",
)]

ITEMS = [g for g in load_golden() if g["category"] == "named_query"]


def _norm(v):
    return v.lower() if isinstance(v, str) else v


def _row_matches(row: dict, want: dict) -> bool:
    return all(_norm(row.get(k)) == _norm(v) for k, v in want.items())


@pytest.mark.parametrize("item", ITEMS, ids=[g["id"] for g in ITEMS])
def test_named_query_item(item):
    _, role = IDENTITIES[item["identity"]]
    sf = snowlane.connect(role=role)
    try:
        for spec, expect in zip(item["queries"], item["expect"], strict=True):
            params = dict(spec)
            name = params.pop("query_name")
            result = snowlane.run_named_query(sf, name, params)
            assert result["status"] == "ok", result
            rows = [dict(zip(result["columns"], r)) for r in result["rows"]]
            assert len(rows) >= expect.get("rows_min", 1), (name, len(rows))
            if "match" in expect:
                assert _row_matches(rows[0], expect["match"]), (name, rows[0])
            if "any" in expect:
                assert any(_row_matches(r, expect["any"]) for r in rows), (name, expect["any"])
            if "all" in expect:
                assert all(_row_matches(r, expect["all"]) for r in rows), name
            if "all_prefix" in expect:
                assert all(str(r.get(k, "")).startswith(v) for r in rows for k, v in expect["all_prefix"].items()), name
    finally:
        sf.close()
