"""MCP tool contracts. The server is a thin adapter, so the contracts under
test are: identity is config not parameter; member data is reachable only
through catalog names; unauthorized identities get a boundary, not data."""

import os

import pytest

from raglab import snowlane
from raglab.mcp_server import IDENTITIES, query_member_data


def _fn(tool):
    return getattr(tool, "fn", tool)


def test_identity_map_is_closed():
    assert set(IDENTITIES) == {"public", "employee", "care_team", "actuary"}
    lane2 = {role for _, role in IDENTITIES.values() if role}
    assert lane2 <= set(snowlane.ROLES)


def test_unknown_query_name_refused_with_catalog():
    result = snowlane.run_named_query(None, "raw_sql; DROP TABLE", {})
    assert result["status"] == "unknown_query"
    assert set(result["known_queries"]) == set(snowlane.NAMED_QUERIES)


def test_public_identity_gets_no_member_data(monkeypatch):
    import raglab.mcp_server as server

    monkeypatch.setattr(server, "PERSONA", "public")
    result = _fn(query_member_data)("member_claims_summary", last_name="X")
    assert result["status"] == "not_authorized"


@pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (verified locally, CI is zero-secret)",
)
def test_named_query_runs_under_examiner_role():
    sf = snowlane.connect(role="CLAIMS_EXAMINER")
    try:
        surname = sf.cursor().execute(
            "SELECT LAST_NAME FROM PATIENTS LIMIT 1"
        ).fetchone()[0]
        result = snowlane.run_named_query(
            sf, "member_claims_summary", {"last_name": surname}
        )
        assert result["status"] == "ok" and result["row_count"] >= 1
        assert "CLAIM_LINES" in result["columns"]
    finally:
        sf.close()
