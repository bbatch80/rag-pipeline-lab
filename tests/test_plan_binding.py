"""Phase 3.5 build 2: a member question that names no plan searches the
member's OWN plan — looked up from enrollment by member id when the member
key is bound, never taken from the screen and never guessed by a model. A
named plan wins; no enrollment row leaves the route unchanged."""
import pytest

from raglab import retrieval, router
from raglab.retrieval import Context


def test_enrolled_plan_is_bound_when_the_question_names_none():
    route = router.route("What's their copay for an urgent-care visit?")
    ctx = Context(member_key="k", enrollment={2025: "71-018", 2026: "71-018"})
    bound = retrieval.bind_enrollment_plan(route, ctx)
    assert bound.plan_codes == ("71-018",) and bound.plan_from_enrollment is True
    assert any("from enrollment" in r for r in bound.reasons)


def test_a_named_plan_wins_and_no_enrollment_leaves_the_route_alone():
    named = router.route("What does the High Option cover for urgent care?")
    ctx = Context(member_key="k", enrollment={2026: "71-018"})
    assert retrieval.bind_enrollment_plan(named, ctx).plan_codes == named.plan_codes  # the question's plan, not enrollment's
    assert retrieval.bind_enrollment_plan(named, ctx).plan_from_enrollment is False
    plain = router.route("What's their copay for an urgent-care visit?")
    assert retrieval.bind_enrollment_plan(plain, Context(member_key="k")) is plain  # no enrollment row -> unchanged
    assert retrieval.bind_enrollment_plan(plain, Context()) is plain               # no member -> unchanged


def test_multi_year_questions_bind_each_routed_year_s_plan():
    route = router.route("How did their urgent care copay change from last year?")
    assert set(route.years) == {2025, 2026}
    ctx = Context(member_key="k", enrollment={2025: "71-006", 2026: "71-018"})
    assert set(retrieval.bind_enrollment_plan(route, ctx).plan_codes) == {"71-006", "71-018"}


def test_probe_searches_only_the_members_plan(db, monkeypatch):
    """Under the fixture: three 2026 brochures; the member is enrolled in
    71-018; a member question returns 71-018 chunks only."""
    from test_governance import _seed_tiers
    from raglab import pipeline
    _seed_tiers(db, per_tier=0, embed=True)
    vec = "[" + ",".join(["0.5"] * 1536) + "]"
    for code in ("71-006", "71-018", "71-021"):
        doc = db.execute("INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) VALUES (%s, %s, 'h', 'public', 1) RETURNING id",
                         (f"t/{code}.pdf", f"GEHA {code} 2026")).fetchone()[0]
        db.execute("INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, plan_code, doc_type, embedding) "
                   "VALUES (%s, 0, %s, 'public', 2026, %s, 'brochure', %s)", (doc, f"urgent care copay {code}", code, vec))
    db.execute("SET LOCAL enable_indexscan = off")
    monkeypatch.setattr("raglab.retrieval.embed_query", lambda t: vec)
    monkeypatch.setattr(retrieval, "_enrollment_plans", lambda conn, key: {2026: "71-018"})
    import raglab.rerank as rr

    class _M:
        def predict(self, pairs): return [0.9] * len(pairs)
    monkeypatch.setattr(rr, "_model", _M())
    ctx = retrieval.Context(member_key="00000000-0000-4000-8000-000000000001", enrollment={2026: "71-018"})
    from raglab.timing import Stopwatch
    probe = pipeline._probe(db, "urgent care copay", "public", ctx, Stopwatch())
    assert probe.decision.plan_codes == ("71-018",) and probe.decision.plan_from_enrollment
    assert {c.plan_code for c in probe.reranked} == {"71-018"}


@pytest.mark.readonly
def test_resolver_reads_the_members_enrollment_from_the_real_table(db):
    """Against the live enrollment table (skipped where the Synthea schema is
    absent): the fake in the probe test cannot hide a broken lookup — the
    first version compared a text column to a uuid and silently found nothing."""
    try:
        row = db.execute("SELECT member_id, patient FROM synthea.enrollment WHERE plan_code IS NOT NULL LIMIT 1").fetchone()
    except Exception:  # noqa: BLE001
        pytest.skip("no synthea.enrollment here")
    member_id, patient = row
    ctx = retrieval.resolve_context(db, member_id, "What is their urgent care copay?")
    assert ctx.member_key == str(patient)
    assert ctx.enrollment and all(isinstance(y, int) and code for y, code in ctx.enrollment.items())
    bound = retrieval.bind_enrollment_plan(router.route("What is their urgent care copay?"), ctx)
    assert bound.plan_from_enrollment and bound.plan_codes


def test_composer_binds_the_enrolled_plan_for_warehouse_slots(db, monkeypatch):
    """Probe 6: a warehouse leg whose slot is plan_code fills it from the
    member's enrollment, the same binding the document legs get."""
    from raglab import planner
    from test_governance import _seed_tiers
    from test_identical_question_control import _NoCommit, _exact_scan, _fake_models
    _seed_tiers(db, per_tier=1, embed=True); _fake_models(monkeypatch); _exact_scan(db)
    monkeypatch.setattr(planner.retrieval, "resolve_context",
                        lambda conn, member_id, q: retrieval.Context(query=q, member_key="k", record={"member_id": member_id}, enrollment={2026: "71-018"}))
    seen = {}

    class _SF:
        def cursor(self): return self
        def execute(self, sql, params=None):
            if isinstance(params, dict): seen["bound"] = params
            self.description = [("PROVIDER_NAME",)]; return self
        def fetchone(self): return ("MEMBER_SERVICES_REP",)
        def fetchall(self): return [("Dr A",)]
        def close(self): pass

    plan = planner.plan_from_dict({"shape": "compound", "legs": [
        {"name": "enrollment", "kind": "member_query", "query_name": "member_enrollment", "slots": ["member_id"]},
        {"name": "network", "kind": "member_query", "query_name": "providers_by_specialty", "slots": ["plan_code"]}]})
    out = planner.compose(_NoCommit(db), "Which doctors are in network for this member?", planner.Caller(persona="member_services", warehouse_role="MEMBER_SERVICES_REP"),
                          member_id="M982263550", plan=plan, module="agent_assist", sf_connect=lambda role: _SF())
    assert out["status"] == "ok" and seen["bound"]["plan_code"] == "71-018"
    assert out["router"]["plan_from_enrollment"] is True
