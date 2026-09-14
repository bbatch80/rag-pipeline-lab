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
    ({"legs": []}, "1..4 legs"),
    ({"legs": [{"name": f"l{i}", "kind": "doc_probe", "text": "q"} for i in range(5)]}, "1..4 legs"),
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
                        lambda conn, member_id, q, case_id=None: Context(query=q, record={"claim_id": "CLM-1363781509"}, as_of="2024-08-01"))
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
    assert "member_appeals" in planner.MODULES["agent_assist"]["named_queries"]  # appeal status is ordinary member service (2026-09-14)
    assert "appeal_case" not in planner.MODULES["agent_assist"]["named_queries"]   # working a case stays on the workbench


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


def test_out_of_scope_route_short_circuits_composition(db, monkeypatch):
    """No plan, no legs, status out_of_scope with the boundary text at the
    top (scope_negative-01, 2026-09-12)."""
    from raglab import planner, router
    stored = router.Reading(scope="other_carrier", boundary_value="Blue Cross FEP", origin="model")
    monkeypatch.setattr(planner, "PLANNER", "model")
    monkeypatch.setattr(planner, "_read_prepare", lambda conn, q: (("k",), q, stored))  # the model's reading, from the store
    monkeypatch.setattr(planner, "_disclose_and_commit", lambda *a, **k: None)
    built = planner.compose(db, "What does Blue Cross FEP Basic charge for a specialist visit?", planner.Caller(persona="public"), module="ask", source="test")
    assert built["status"] == "out_of_scope" and "Blue Cross FEP" in built["boundary_response"]
    assert built["chunks"] == [] and built["plan"] is None and built["sub_results"] == [] and built["warehouse_results"] == []


def test_a_question_about_what_a_document_said_gets_a_document_leg():
    """L2 (appeal-03, persona_negative-03, the Workbench probe): the model's
    warehouse-only plan gains one document leg over the module's menu; plans
    that already search documents, caller plans, and full plans are untouched."""
    from raglab import planner

    def plan(*legs, origin="model"):
        return planner.Plan(shape="simple" if len(legs) == 1 else "compound", origin=origin,
                            legs=[planner.Leg(**l) for l in legs])

    wh = {"name": "case", "kind": "member_query", "query_name": "appeal_case", "slots": ("case_id",)}
    out = planner.enforce_document_leg(plan(wh), "What did the determination letter tell the member on case APL-1215086?")
    assert [l.kind for l in out.legs] == ["member_query", "doc_probe"] and out.enforced == ("document_leg",)
    assert out.legs[1].sources == () and out.shape == "compound"        # the module's whole document menu
    out = planner.enforce_document_leg(plan(wh), "Why was the denial upheld?")
    assert out.enforced == ("document_leg",)
    assert planner.enforce_document_leg(plan(wh), "When did the member file the appeal?").enforced == ()  # a fact: no rule
    doc = {"name": "d", "kind": "doc_probe", "text": "x"}
    assert planner.enforce_document_leg(plan(wh, doc), "What did the letter say?").enforced == ()    # already searches
    assert planner.enforce_document_leg(plan(wh, origin="caller"), "What did the letter say?").enforced == ()
    full = plan(wh, {**wh, "name": "b"}, {**wh, "name": "c"}, {**wh, "name": "d"})
    assert planner.enforce_document_leg(full, "What did the letter say?").enforced == ()              # at the leg limit


def test_an_open_ended_benefits_question_becomes_fact_shaped_legs():
    """Run 667's lesson: a rule that fires only on this shape. 'How does
    mental-health coverage work?' → cost / coverage / limits legs over the
    same family; a fact question, a records leg, or a caller plan is untouched."""
    from raglab import planner

    def plan(*legs, origin="model"):
        return planner.Plan(shape="simple" if len(legs) == 1 else "compound", origin=origin, legs=[planner.Leg(**l) for l in legs])

    doc = {"name": "d", "kind": "doc_probe", "text": "How does mental health coverage work? What are the copays", "sources": ("brochure",)}
    out = planner.expand_open_ended_legs(plan(doc), "How does mental-health coverage work?")
    assert out.enforced == ("fact_shaped_legs",) and len(out.legs) == 3 and out.shape == "compound"
    assert [l.text for l in out.legs][0] == "What is the copay or coinsurance for mental-health?"
    assert all(l.kind == "doc_probe" and l.sources == ("brochure",) for l in out.legs) and out.legs[2].required is False
    for q in ("What does the plan cover for hearing aids?", "Tell me about the dental benefit", "How is chiropractic care covered?"):
        assert planner.expand_open_ended_legs(plan(doc), q).enforced == ("fact_shaped_legs",), q
    fact = {"name": "d", "kind": "doc_probe", "text": "specialist copay on the High Option", "sources": ("brochure",)}
    assert planner.expand_open_ended_legs(plan(fact), "What is the specialist copay on the High Option?").enforced == ()
    note = {"name": "n", "kind": "doc_probe", "text": "clinical history", "sources": ("clinical_note",)}
    assert planner.expand_open_ended_legs(plan(note), "How does her clinical history work?").enforced == ()  # records, not benefits
    assert planner.expand_open_ended_legs(plan(doc, origin="caller"), "How does mental-health coverage work?").enforced == ()
    wh = {"name": "c", "kind": "member_query", "query_name": "appeal_case", "slots": ("case_id",)}
    both = planner.expand_open_ended_legs(plan(wh, doc), "How does mental-health coverage work?")
    assert [l.kind for l in both.legs] == ["member_query", "doc_probe", "doc_probe", "doc_probe"]  # room for three (limit 4)


def test_every_plan_coverage_applies_only_to_legs_that_reach_brochures():
    """Run 669: a bulletin or formulary leg under 'nothing named' coverage
    searched six plans' brochures beside it and lost its seats."""
    from raglab import planner, retrieval, router

    route = router.route("Which timely-filing bulletin was in force in June 2025?")
    assert route.cover_level == "all" and len(route.plan_codes) > 1
    ctx = retrieval.Context(query="q")
    available = {"module": "agent_assist", "sources": ("bulletin", "brochure", "kb"), "named_queries": ()}
    bulletin = planner.Leg(name="b", kind="doc_probe", text="timely filing", sources=("bulletin",))
    lr = planner._leg_route(bulletin, route, ctx, available)
    assert lr.plan_codes == () and lr.cover_level is None and any("not applied" in r for r in lr.reasons)
    brochure = planner.Leg(name="d", kind="doc_probe", text="deductible", sources=("brochure",))
    assert planner._leg_route(brochure, route, ctx, available).cover_level == "all"
    unhinted = planner.Leg(name="u", kind="doc_probe", text="deductible", sources=())
    assert planner._leg_route(unhinted, route, ctx, available).cover_level == "all"


def test_a_list_of_benefits_becomes_one_fact_shaped_leg_each():
    """The user's four-topic cost question: one leg per named benefit (limit
    four); 'does the plan cover A, B and C' likewise; single-benefit and
    already-reshaped plans are untouched."""
    from raglab import planner

    def plan(*legs, origin="model", enforced=()):
        pl = planner.Plan(shape="simple" if len(legs) == 1 else "compound", origin=origin, legs=[planner.Leg(**l) for l in legs])
        pl.enforced = enforced
        return pl

    doc = {"name": "d", "kind": "doc_probe", "text": "costs for physical therapy, imaging, outpatient surgery, and specialist visits", "sources": ("brochure",)}
    verbatim = lambda items: None  # noqa: E731 — no row titles: the member's words make the legs
    q4 = "What are the costs for physical therapy, imaging, outpatient surgery, and specialist visits?"
    out = planner.split_benefit_list(plan(doc), q4, titles=verbatim)
    assert out.enforced == ("split_by_benefit",) and len(out.legs) == 4 and out.shape == "compound"
    assert [l.text for l in out.legs] == ["What do I pay for physical therapy?", "What do I pay for imaging?",
                                          "What do I pay for outpatient surgery?", "What do I pay for specialist visits?"]
    assert [l.name for l in out.legs] == ["physical therapy: cost", "imaging: cost", "outpatient surgery: cost", "specialist visits: cost"]
    # with the brochure's row titles (the reranker scores a row near 1 against its own title, near 0 against a synonym)
    rows = lambda items: ("Physical, occupational, speech, habilitative and rehabilitative therapy", "Lab, x-ray and other diagnostic tests",  # noqa: E731
                          "Outpatient hospital or ambulatory surgical center", "Physician office visits")
    titled = planner.split_benefit_list(plan(doc), q4, titles=rows)
    assert titled.legs[1].text == "What do I pay for Lab, x-ray and other diagnostic tests?" and titled.legs[1].name == "imaging: cost"
    named = planner.split_benefit_list(plan(doc), "What are the copays for urgent care and telehealth on the High Option?", titles=verbatim)
    assert [l.text for l in named.legs] == ["What do I pay for urgent care on the High Option?", "What do I pay for telehealth on the High Option?"]
    cover = planner.split_benefit_list(plan(doc), "Does the plan cover acupuncture, chiropractic care, and massage therapy?", titles=verbatim)
    assert [l.text for l in cover.legs][0] == "Does the plan cover acupuncture?" and len(cover.legs) == 3
    assert planner.split_benefit_list(plan(doc), "What is the copay for a specialist visit?", titles=verbatim).enforced == ()        # one benefit
    assert planner.split_benefit_list(plan(doc, enforced=("fact_shaped_legs",)), "costs for a, b, and c", titles=verbatim).enforced == ("fact_shaped_legs",)
    five = "What are the costs for a, b, c, d, and e?"
    assert planner.split_benefit_list(plan(doc), five, titles=verbatim).enforced == ()                                             # over the limit: untouched
    wh = {"name": "c", "kind": "member_query", "query_name": "appeal_case", "slots": ("case_id",)}
    assert len(planner.split_benefit_list(plan(wh, doc), q4, titles=verbatim).legs) == 2  # no room: untouched


def test_row_titles_come_from_the_model_and_fall_back_to_the_members_words():
    from raglab import planner

    class _Resp:
        def __init__(self, text):
            self.content = [type("B", (), {"type": "text", "text": text})()]

    class _Client:
        def __init__(self, text):
            self.text, self.calls = text, []

        @property
        def messages(self):
            return self

        def create(self, **kw):
            self.calls.append(kw)
            return _Resp(self.text)

    planner.benefit_row_titles.cache_clear()
    good = _Client('["Lab, x-ray and other diagnostic tests", "Physician office visits"]')
    assert planner.benefit_row_titles(("imaging", "specialist visits"), client=good) == ("Lab, x-ray and other diagnostic tests", "Physician office visits")
    assert good.calls[0]["extra_body"] == {"temperature": 0.0} and good.calls[0]["model"] == planner.PLANNER_MODEL
    assert planner.benefit_row_titles(("imaging",), client=_Client("[\"one\", \"two\"]")) is None      # wrong count: the member's words
    assert planner.benefit_row_titles(("imaging",), client=_Client("not json")) is None
    planner.benefit_row_titles.cache_clear()


def test_an_invented_query_name_snaps_to_the_menu_or_loses_only_its_leg():
    """The user's live probe 2026-09-14: 'member_appeals_history' failed
    validation and the whole plan fell back to rules."""
    from raglab import planner

    menu = {"sources": ("brochure",), "named_queries": ("member_profile", "member_appeals", "member_calls")}
    legs = [planner.Leg(name="a", kind="member_query", query_name="member_appeals_history", slots=("member_id",)),
            planner.Leg(name="d", kind="doc_probe", text="appeal rights", sources=("brochure",))]
    pl = planner.repair_query_names(planner.Plan(shape="compound", origin="model", legs=legs), menu)
    assert pl.legs[0].query_name == "member_appeals" and pl.enforced == ("query_name:member_appeals_history->member_appeals",)
    lost = [planner.Leg(name="z", kind="member_query", query_name="zebra_report", slots=()), legs[1]]
    pl = planner.repair_query_names(planner.Plan(shape="compound", origin="model", legs=lost), menu)
    assert [l.name for l in pl.legs] == ["d"] and pl.shape == "simple" and pl.enforced == ("dropped:zebra_report",)
    clean = planner.repair_query_names(planner.Plan(shape="simple", origin="model", legs=[legs[1]]), menu)
    assert clean.enforced == ()
    assert "member_appeals" in planner.MODULES["agent_assist"]["named_queries"] and "member_appeals" in planner.MODULES["care_management"]["named_queries"]


def test_a_member_scoped_query_always_binds_its_member():
    """named_query-12 (2026-09-14): the model planned member_appeals with no
    slots; the query ran unbound and reported 'no rows'."""
    from raglab import planner

    legs = [planner.Leg(name="a", kind="member_query", query_name="member_appeals", slots=()),
            planner.Leg(name="p", kind="member_query", query_name="provider_lookup", slots=(), params={"npi": "1"}),
            planner.Leg(name="c", kind="member_query", query_name="member_calls", slots=("member_id",))]
    pl = planner.fill_member_slot(planner.Plan(shape="compound", origin="model", legs=legs))
    assert pl.legs[0].slots == ("member_id",) and pl.enforced == ("member_slot:member_appeals",)
    assert pl.legs[1].slots == () and pl.legs[2].slots == ("member_id",)  # no member_id in the catalog; already bound
    assert planner._DOCUMENT_WORDS.search("what did they call about immediately before the call where they disputed a denial?")  # call_note-08


def test_a_platform_added_document_leg_is_best_effort():
    """'Can you tell me if Cesar has any recent calls with us?' (2026-09-14):
    the model's calls query found the row; the rule-added document leg found
    nothing and, being required, hid the answer. Added legs are never
    required, and 'tell me' is not a request about documents."""
    from raglab import payload as payload_mod
    from raglab import planner, router

    assert not planner._DOCUMENT_WORDS.search("Can you tell me if Cesar has any recent calls with us?")
    assert planner._DOCUMENT_WORDS.search("What did we tell them when they called about the denied claim?")
    assert planner._DOCUMENT_WORDS.search("What did the rep tell the member?")
    plan = planner.Plan(shape="simple", origin="model", legs=[planner.Leg(name="c", kind="member_query", query_name="member_calls", slots=("member_id",))])
    out = planner.enforce_document_leg(plan, "What did the letter say about the appeal?")
    assert out.enforced == ("document_leg",) and out.legs[-1].kind == "doc_probe" and out.legs[-1].required is False
    route = router.Route(scope="in_scope", years=(2026,))
    composed = payload_mod.compose("q", out.to_dict(),
                                   sub_results=[{"leg": "documents", "status": "insufficient_evidence", "confidence": 0.003, "chunk_indexes": []}],
                                   warehouse_results=[{"leg": "c", "query_name": "member_calls", "status": "ok", "rows": [["C1"]], "columns": ["CALL_ID"], "row_count": 1}],
                                   chunks=[], subject="M1", unresolved=[], as_of_defaulted=False)
    assert composed["status"] == "ok" and composed["missing"] == []   # the model's leg answered; the added leg's silence does not sink it


def test_a_member_leg_with_no_member_open_is_released_not_failed():
    """A member's own question on Ask (2026-09-14): 'will my old plan cover
    January 7 … does my deductible reset?' The brochure note answered at
    0.77; the planner's enrollment leg could not bind (no member on Ask),
    was required, and hid the answer behind 'missing: current_enrollment'."""
    from raglab import planner

    legs = [planner.Leg(name="current_enrollment", kind="member_query", query_name="member_enrollment", slots=("member_id",)),
            planner.Leg(name="docs", kind="doc_probe", text="when does a new plan take effect", sources=("brochure",))]
    pl = planner.release_unbound_member_legs(planner.Plan(shape="compound", origin="model", legs=legs), member_id=None)
    assert pl.legs[0].required is False and pl.legs[1].required is True and pl.enforced == ("no_member:member_enrollment",)
    legs2 = [planner.Leg(name="current_enrollment", kind="member_query", query_name="member_enrollment", slots=("member_id",))]
    with_member = planner.release_unbound_member_legs(planner.Plan(shape="simple", origin="model", legs=legs2), member_id="M1")
    assert with_member.legs[0].required is True and with_member.enforced == ()   # a member is open: the leg runs as planned
    provider = [planner.Leg(name="p", kind="member_query", query_name="provider_lookup", slots=(), params={"npi": "1"})]
    assert planner.release_unbound_member_legs(planner.Plan(shape="simple", origin="model", legs=provider), None).enforced == ()  # not member-scoped

def test_one_question_per_document_leg():
    """The transition question: one leg holding two questions scored the
    Open Season note 0.12; either question alone 0.41-0.77. Split at the
    question boundary; the first keeps required, the rest are best-effort."""
    from raglab import planner

    leg = planner.Leg(name="rules", kind="doc_probe", sources=("brochure",),
                      text="When a member switches plans at year-end, does the old plan cover services before the new plan takes effect? Does the deductible reset when switching plans?")
    out = planner.split_multi_question_leg(planner.Plan(shape="simple", origin="model", legs=[leg]))
    assert [l.text for l in out.legs] == ["When a member switches plans at year-end, does the old plan cover services before the new plan takes effect?",
                                          "Does the deductible reset when switching plans?"]
    assert [l.required for l in out.legs] == [True, False] and out.enforced == ("split_by_question",) and out.shape == "compound"
    assert planner.split_multi_question_leg(out) is out                                                    # idempotent
    single = planner.Plan(shape="simple", origin="model", legs=[planner.Leg(name="d", kind="doc_probe", text="What is the deductible? ", sources=())])
    assert planner.split_multi_question_leg(single).enforced == ()                                        # one question: untouched
    many = planner.Plan(shape="simple", origin="model", legs=[planner.Leg(name="d", kind="doc_probe", text="A? B? C? D? E?", sources=())])
    assert planner.split_multi_question_leg(many).enforced == ()                                          # over the leg limit: untouched


def test_brochure_vocabulary_sets_the_ranking_text_and_keeps_the_wording():
    from raglab import planner

    legs = [planner.Leg(name="a", kind="doc_probe", text="Does the deductible reset when switching plans?", sources=("brochure",)),
            planner.Leg(name="c", kind="doc_probe", text="what did the rep say", sources=("call_note",)),
            planner.Leg(name="m", kind="member_query", query_name="member_calls", slots=("member_id",))]
    phrasing = lambda texts: tuple("Is the calendar year deductible met again from January 1?" for _ in texts)  # noqa: E731
    out = planner.phrase_legs_in_brochure_vocabulary(planner.Plan(shape="compound", origin="model", legs=legs), phrasing=phrasing)
    assert out.legs[0].ranking_text == "Is the calendar year deductible met again from January 1?" and out.legs[0].text.startswith("Does the deductible")
    assert out.legs[1].ranking_text is None and out.legs[2].ranking_text is None    # records legs and warehouse legs are left alone
    assert out.enforced == ("vocabulary",)
    stored = planner.plan_from_dict(out.to_dict(), origin="model", model="m")
    assert stored.legs[0].ranking_text == out.legs[0].ranking_text                    # travels with the stored plan
    fresh = planner.Leg(name="a", kind="doc_probe", text="Does the deductible reset when switching plans?", sources=("brochure",))
    untouched = planner.phrase_legs_in_brochure_vocabulary(planner.Plan(shape="simple", origin="model", legs=[fresh]), phrasing=lambda t: None)
    assert untouched.enforced == () and untouched.legs[0].ranking_text is None        # the call failed: the member's words rank
