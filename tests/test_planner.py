"""Phase 3 planner: plans are fixed-menu objects validated before execution,
executed as the caller's identity, composed worst-of-required, disclosed
once. Offline where possible; database tests use the rolled-back fixture
with seeded tiers and fake models (see test_identical_question_control)."""
import pytest

from raglab import planner
from raglab.planner import Caller, Leg, Plan, PlanError
from test_governance import _seed_tiers
from test_identical_question_control import _NoCommit, _exact_scan, _fake_models

MENU = {"sources": ("brochure", "sop", "clinical_note", "call_note", "appeal", "clinical_policy"),
        "named_queries": ("claim_adjudication", "member_calls", "appeal_case")}
Q = "Was claim CLM-1363781509 for member M344317862 denied, and did the member appeal it?"


def _plan(*legs, shape=None):
    return planner.plan_from_dict({"shape": shape, "legs": list(legs)})


# ---- validation: nothing outside the menu, nothing the question did not say

def test_rules_plan_is_one_document_leg_on_the_original_text():
    p = planner.plan_rules(Q)
    assert p.shape == "simple" and p.origin == "rules" and len(p.legs) == 1
    assert p.legs[0].kind == "doc_probe" and p.legs[0].text == Q and p.legs[0].sources == ()
    planner.validate(p, Q, MENU)


@pytest.mark.parametrize("bad, message", [
    ({"legs": []}, "1..3 legs"),
    ({"legs": [{"name": f"l{i}", "kind": "doc_probe", "text": "q"} for i in range(4)]}, "1..3 legs"),
    ({"legs": [{"name": "a", "kind": "doc_probe", "text": "q"}, {"name": "a", "kind": "doc_probe", "text": "q"}]}, "unique"),
    ({"legs": [{"name": "x", "kind": "sql", "text": "q"}]}, "unknown kind"),
    ({"legs": [{"name": "x", "kind": "doc_probe", "text": "q", "sources": ["payroll"]}]}, "not on the menu"),
    ({"legs": [{"name": "x", "kind": "doc_probe", "text": "appeal APL-9999999 status"}]}, "identifier not in the question"),
    ({"legs": [{"name": "x", "kind": "doc_probe", "text": "status of member M999900004"}]}, "identifier not in the question"),
    ({"legs": [{"name": "x", "kind": "member_query", "query_name": "drop_tables"}]}, "not in the catalog"),
    ({"legs": [{"name": "x", "kind": "member_query", "query_name": "member_calls", "params": {"member_id": "M344317862"}}]}, "bound from context"),
    ({"legs": [{"name": "x", "kind": "member_query", "query_name": "member_calls", "params": {"where": "1=1"}}]}, "not accepted"),
])
def test_invalid_plans_are_rejected_before_execution(bad, message):
    with pytest.raises(PlanError, match=message):
        planner.validate(planner.plan_from_dict(bad), Q, MENU)


def test_a_leg_may_repeat_identifiers_the_question_contains():
    p = _plan({"name": "appeal", "kind": "doc_probe", "text": "Did member M344317862 appeal claim CLM-1363781509?", "sources": ["appeal"]},
              {"name": "adj", "kind": "member_query", "query_name": "claim_adjudication", "slots": ["claim_id"]})
    planner.validate(p, Q, MENU)
    assert p.shape == "compound"


# ---- slots bind from context only

def test_slots_bind_from_resolved_context_never_from_the_plan():
    from raglab.retrieval import Context
    from raglab.router import Route
    leg = Leg(name="adj", kind="member_query", query_name="claim_adjudication", slots=("claim_id", "member_id"))
    ctx = Context(record={"claim_id": "CLM-1363781509"})
    bound = planner.bind_slots(leg, ctx, "M344317862", Route(scope="in_scope"))
    assert bound == {"claim_id": "CLM-1363781509", "member_id": "M344317862"}
    with pytest.raises(PlanError, match="member_id required but not in context"):
        planner.bind_slots(leg, Context(record={"claim_id": "CLM-1363781509"}), None, Route(scope="in_scope"))


# ---- execution under the database fixture

def _caller(persona, role=None):
    return Caller(persona=persona, warehouse_role=role)


def test_fast_path_matches_the_v1_probe_and_discloses_once(db, monkeypatch):
    from raglab.pipeline import run_query
    _seed_tiers(db, per_tier=1, embed=True); _fake_models(monkeypatch); _exact_scan(db)
    before = db.execute("SELECT count(*) FROM disclosure_log").fetchone()[0]
    v1 = run_query(_NoCommit(db), "what do the secret facts say", persona="employee")
    db.execute("RESET ROLE")
    composed = planner.compose(_NoCommit(db), "what do the secret facts say", _caller("employee"))
    db.execute("RESET ROLE")
    assert composed["status"] == v1["status"] == "ok"
    assert {c["source"]["title"] for c in composed["chunks"]} == {c["source"]["title"] for c in v1["chunks"]}
    assert composed["plan"]["origin"] == "rules" and composed["plan"]["shape"] == "simple" and composed["missing"] == []
    assert composed["spec_version"] == "1.1.0" and "timings" in composed
    assert db.execute("SELECT count(*) FROM disclosure_log").fetchone()[0] == before + 2  # one row per question, not per leg


def test_worst_of_required_composition_and_identity_passthrough(db, monkeypatch):
    """A warehouse leg with no warehouse identity is not executed; if it is
    required the composed status is insufficient_evidence and missing[] names
    it, while the document leg's evidence still rides in the payload."""
    _seed_tiers(db, per_tier=1, embed=True); _fake_models(monkeypatch); _exact_scan(db)
    plan = _plan({"name": "docs", "kind": "doc_probe", "text": "what do the secret facts say"},
                 {"name": "adj", "kind": "member_query", "query_name": "claim_adjudication", "slots": ["claim_id"]})
    out = planner.compose(_NoCommit(db), "what do the secret facts say about claim CLM-1363781509", _caller("employee"), plan=plan,
                          sf_connect=lambda role: pytest.fail("must not open a warehouse session without a role"))
    db.execute("RESET ROLE")
    assert out["status"] == "insufficient_evidence" and out["missing"] == ["adj"]
    assert out["sub_results"][0]["status"] == "ok" and len(out["chunks"]) >= 1
    assert out["warehouse_results"][0]["status"] == "not_executed"
    # the same leg marked optional never forces a refusal
    plan.legs[1].required = False
    out = planner.compose(_NoCommit(db), "what do the secret facts say about claim CLM-1363781509", _caller("employee"), plan=plan)
    db.execute("RESET ROLE")
    assert out["status"] == "ok" and out["missing"] == []


def test_warehouse_leg_runs_as_the_callers_role_with_bound_parameters(db, monkeypatch):
    _seed_tiers(db, per_tier=1, embed=True); _fake_models(monkeypatch); _exact_scan(db)
    seen = {}

    class _SF:
        def cursor(self):
            return self
        def execute(self, sql, params=None):
            seen.setdefault("sql", []).append((sql, params)); self.description = [("CLAIM_ID",), ("STATUS",)]; return self
        def fetchone(self):
            return ("APPEALS_ANALYST",)
        def fetchall(self):
            return [("CLM-1363781509", "denied")]
        def close(self):
            seen["closed"] = True

    def connect(role):
        seen["role"] = role; return _SF()

    # The claim resolves through the Synthea population, which CI's fresh
    # database does not load: supply the resolved context so the test checks
    # the binding path itself, not the population.
    from raglab.retrieval import Context
    monkeypatch.setattr(planner.retrieval, "resolve_context",
                        lambda conn, member_id, q: Context(query=q, record={"claim_id": "CLM-1363781509"}, as_of="2024-08-01"))
    plan = _plan({"name": "adj", "kind": "member_query", "query_name": "claim_adjudication", "slots": ["claim_id"]})
    out = planner.compose(_NoCommit(db), "Was claim CLM-1363781509 denied?", _caller("appeals", "APPEALS_ANALYST"), plan=plan, sf_connect=connect)
    assert seen["role"] == "APPEALS_ANALYST" and seen["closed"]
    bound = [p for s, p in seen["sql"] if isinstance(p, dict)][0]
    assert bound["claim_id"] == "CLM-1363781509"
    assert out["status"] == "ok" and out["warehouse_results"][0]["row_count"] == 1
    assert out["chunks"] == [] and out["plan"]["shape"] == "simple"
    assert out["record_context"]["as_of"] == "2024-08-01"  # decision 5: bound before planning, from the lookup


def test_module_menu_bounds_the_plan_and_hints_never_filter(db, monkeypatch):
    """Decision 7: the planner chooses only from the module's menu — an
    off-menu query is rejected before execution — and a document leg searches
    the module's whole document menu: a source hint is recorded, never a
    filter, so a wrongly hinted leg still finds evidence in another family."""
    _seed_tiers(db, per_tier=1, embed=True); _fake_models(monkeypatch); _exact_scan(db)
    from raglab import pipeline
    routes = []
    real = pipeline._probe

    def spy(conn, query, persona, ctx, watch, decision=None):
        routes.append(tuple(decision.sources) if decision else None); return real(conn, query, persona, ctx, watch, decision)
    monkeypatch.setattr(planner, "_probe", spy)
    # hinted at a family the employee cannot see, inside the 'ask' module: the leg searches ask's whole menu and still answers
    plan = _plan({"name": "docs", "kind": "doc_probe", "text": "what do the secret facts say", "sources": ["clinical_policy"]})
    out = planner.compose(_NoCommit(db), "what do the secret facts say", _caller("employee"), plan=plan, module="ask")
    db.execute("RESET ROLE")
    assert out["status"] == "ok" and out["plan"]["widened"] is False and out["plan"]["module"] == "ask"
    assert set(routes[-1]) >= {"brochure", "sop"} and "call_note" not in routes[-1], "the module's document menu, not the hint"
    # an off-menu query for the module is rejected before anything runs
    plan = _plan({"name": "x", "kind": "member_query", "query_name": "appeal_case", "slots": ["case_id"]})
    with pytest.raises(PlanError, match="not in the catalog"):
        planner.compose(_NoCommit(db), "What did appeal APL-1020254 decide?", _caller("member_services", "MEMBER_SERVICES_REP"), plan=plan, module="agent_assist")
    with pytest.raises(PlanError, match="unknown module"):
        planner.compose(_NoCommit(db), "q", _caller("employee"), module="payroll")


def test_modules_are_a_closed_menu_over_the_catalog():
    from raglab import snowlane
    for name, spec in planner.MODULES.items():
        assert set(spec["named_queries"]) <= set(snowlane.NAMED_QUERIES), name
        assert set(spec["sources"]) <= set(planner.SOURCE_GUIDE), name
    assert planner.MODULES["ask"]["named_queries"] == ()  # Ask never reaches the warehouse
    assert "member_appeals" not in planner.MODULES["agent_assist"]["named_queries"]  # the rep has no grant on APPEALS


def test_compose_records_every_stage_in_order(db, monkeypatch):
    _seed_tiers(db, per_tier=1, embed=True); _fake_models(monkeypatch); _exact_scan(db)
    events = []
    plan = _plan({"name": "docs", "kind": "doc_probe", "text": "what do the secret facts say"},
                 {"name": "adj", "kind": "member_query", "query_name": "claim_adjudication", "slots": ["claim_id"]})
    planner.compose(_NoCommit(db), "what do the secret facts say about claim CLM-1363781509", _caller("employee"), plan=plan, module="agent_assist", trace=events)
    db.execute("RESET ROLE")
    assert [e["stage"] for e in events] == ["question", "route", "identifiers", "menu", "plan", "doc_leg", "leg_verdict", "warehouse_leg", "composed", "disclosed"]
    doc = next(e for e in events if e["stage"] == "doc_leg")
    assert doc["vector_top"] and doc["rerank_top"] and set(doc["sources_searched"]) >= {"brochure", "sop"}
    assert events[-1]["payload_id"] and events[-2]["status"] in ("ok", "insufficient_evidence")
