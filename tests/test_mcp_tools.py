"""MCP tool contracts. The server is a thin adapter, so the contracts under
test are: identity is config not parameter; member data is reachable only
through catalog names; unauthorized identities get a boundary, not data."""

import os

import pytest

from raglab import snowlane
from raglab.mcp_server import IDENTITIES, compose_context, query_member_data


def _fn(tool):
    return getattr(tool, "fn", tool)


def test_identity_map_is_closed():
    assert set(IDENTITIES) == {"public", "employee", "care_team", "member_services", "appeals", "actuary"}
    lane2 = {role for _, role in IDENTITIES.values() if role}
    assert lane2 <= set(snowlane.ROLES)


def test_unknown_query_name_refused_with_catalog():
    result = snowlane.run_named_query(None, "raw_sql; DROP TABLE", {})
    assert result["status"] == "unknown_query"
    assert set(result["known_queries"]) == set(snowlane.NAMED_QUERIES)


def test_golden_two_lane_schema():
    """Offline: two-lane golden entries carry both decomposed halves and a
    known identity (the live assertions run in test_two_lane_golden.py)."""
    from raglab.ablation import load_golden

    two_lane = [g for g in load_golden() if g["category"] == "two_lane"]
    assert len(two_lane) == 7
    for item in two_lane:
        assert item["identity"] in IDENTITIES
        assert item["doc_probe"] and item["doc_anchor"]
        assert "query_name" in item["member_query"]


def test_public_identity_gets_no_member_data(monkeypatch):
    import raglab.mcp_server as server

    monkeypatch.setattr(server, "PERSONA", "public")
    result = _fn(query_member_data)("member_claims_summary", last_name="X")
    assert result["status"] == "not_authorized"


@pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="no Snowflake credentials (verified locally, CI is zero-secret)",
)
@pytest.mark.slow
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


def test_compose_context_runs_as_the_session_identity_never_a_parameter(monkeypatch):
    """The third tool is a thin adapter: identity comes from the session,
    the plan from the platform, every leg runs as that identity, and the
    server's one warehouse session is handed over without being closed."""
    import inspect

    import raglab.mcp_server as server
    from raglab import planner

    assert "persona" not in inspect.signature(_fn(compose_context)).parameters
    assert "plan" not in inspect.signature(_fn(compose_context)).parameters
    monkeypatch.setattr(server, "USER", None)
    monkeypatch.setattr(server, "PERSONA", "appeals")
    seen = {}

    def fake_compose(conn, question, caller, member_id=None, plan=None, source="interactive", sf_connect=None):
        seen.update(question=question, caller=caller, member_id=member_id, plan=plan, source=source)
        session = sf_connect("APPEALS_ANALYST")
        session.close()  # a leg closes what it is handed
        seen["session"] = session
        return {"status": "ok", "payload_id": "x"}

    class _Conn:
        def cursor(self):
            return "cursor"
        def close(self):
            raise AssertionError("the shared warehouse session must not be closed by a leg")

    monkeypatch.setattr(planner, "compose", fake_compose)
    monkeypatch.setattr(server, "_snowflake", lambda role: _Conn())
    out = _fn(compose_context)("Was claim CLM-1363781509 denied, and did the member appeal it?", member_id="M344317862")
    assert out["status"] == "ok"
    assert seen["caller"].persona == "appeals" and seen["caller"].warehouse_role == "APPEALS_ANALYST"
    assert seen["member_id"] == "M344317862" and seen["plan"] is None and seen["source"] == "mcp"
    assert seen["session"].cursor() == "cursor"


def test_compose_context_public_identity_has_no_warehouse_role(monkeypatch):
    import raglab.mcp_server as server
    from raglab import planner

    monkeypatch.setattr(server, "USER", None)
    monkeypatch.setattr(server, "PERSONA", "public")
    seen = {}
    monkeypatch.setattr(planner, "compose", lambda conn, q, caller, **kw: seen.update(caller=caller) or {"status": "ok"})
    _fn(compose_context)("What does the HDHP brochure say about copays?")
    assert seen["caller"].persona == "public" and seen["caller"].warehouse_role is None
