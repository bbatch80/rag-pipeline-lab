"""P3-PR2: the pinned planner model — one constrained call for shape + legs,
stored plans reused by key, rules fallback on any model trouble, fallbacks
never reused, and the model sees the translated question. The model is a
fake client here; nothing reaches the network."""
import json

import pytest

from raglab import planner

Q = "Was claim CLM-1363781509 for member M344317862 denied, and did the member appeal it?"
MENU = {"sources": ("brochure", "appeal", "call_note"), "named_queries": ("claim_adjudication", "member_calls"),
        "source_docs": {"brochure": "Plan brochures (public)", "appeal": "Appeal case files (appeals)", "call_note": "Call notes (member services)"}}
ROUTE = {"scope": "in_scope", "boundary_value": None, "program": "none", "options": [], "years": [], "change": False, "as_of": None}
GOOD = {"shape": "compound", "legs": [
    {"name": "adjudication", "kind": "member_query", "text": None, "sources": [], "query_name": "claim_adjudication", "slots": ["claim_id"], "required": True},
    {"name": "appeal", "kind": "doc_probe", "text": "Did member [MEMBER_ID-1] appeal claim [CLAIM_ID-2]?", "sources": ["appeal"], "query_name": None, "slots": [], "required": True}]}


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Response:
    def __init__(self, text): self.content = [_Block(text)]


class FakeClient:
    def __init__(self, *answers):
        self.answers = list(answers); self.calls = []
        self.messages = self
    def create(self, **kw):
        self.calls.append(kw)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer if isinstance(answer, str) else json.dumps(answer))


@pytest.fixture
def translated(monkeypatch):
    """Deterministic tokenization regardless of the vault's contents (CI has none)."""
    monkeypatch.setattr(planner, "translated_for_planning",
                        lambda conn, q: q.replace("CLM-1363781509", "[CLAIM_ID-2]").replace("M344317862", "[MEMBER_ID-1]"))


@pytest.fixture
def store(monkeypatch):
    rows = {}
    monkeypatch.setattr(planner, "PLAN_CACHE", True)
    monkeypatch.setattr(planner, "_stored_plan", lambda conn, key: rows.get(key) if rows.get(key, {}).get("origin") == "model" else None)
    def put(conn, key, translated, plan, reason, latency):
        if rows.get(key, {}).get("origin") == "model":
            return
        rows[key] = {"plan": plan.to_dict(), "origin": plan.origin, "reason": reason}
    monkeypatch.setattr(planner, "_store_plan", put)
    return rows


def test_model_plan_is_validated_stored_and_reused(translated, store):
    client = FakeClient(GOOD)
    plan = planner.plan_with_model(None, Q, MENU, client=client)
    assert plan.origin == "model" and plan.shape == "compound" and [l.kind for l in plan.legs] == ["member_query", "doc_probe"]
    assert plan.fallback_reason is None and plan.model == planner.PLANNER_MODEL
    assert len(store) == 1 and next(iter(store.values()))["origin"] == "model"
    again = planner.plan_with_model(None, Q, MENU, client=FakeClient(RuntimeError("must not be called")))
    assert again.to_dict()["legs"] == plan.to_dict()["legs"] and again.origin == "model"


def test_model_sees_the_translated_question_and_the_menu_never_raw_identifiers(translated, store):
    client = FakeClient(GOOD)
    planner.plan_with_model(None, Q, MENU, client=client)
    kw = client.calls[0]
    user = kw["messages"][0]["content"]
    assert "[CLAIM_ID-2]" in user and "[MEMBER_ID-1]" in user
    assert "CLM-1363781509" not in user and "M344317862" not in user
    assert "claim_adjudication" in user and "Appeal case files" in user
    assert kw["model"] == planner.PLANNER_MODEL and kw["output_config"]["format"]["type"] == "json_schema"
    assert "maxItems" not in json.dumps(kw["output_config"])  # the API rejects array bounds; validate() enforces 1..3


@pytest.mark.parametrize("answer, reason", [
    ("not json at all", "JSONDecodeError"),
    ({"shape": "compound", "legs": [{"name": f"l{i}", "kind": "doc_probe", "text": "q", "sources": [], "query_name": None, "slots": [], "required": True} for i in range(4)]}, "PlanError"),
    ({"shape": "simple", "legs": [{"name": "x", "kind": "member_query", "text": None, "sources": [], "query_name": "drop_tables", "slots": [], "required": True}]}, "PlanError"),
    ({"shape": "simple", "legs": [{"name": "x", "kind": "doc_probe", "text": "appeal [CASE_ID-9] APL-9999999", "sources": ["appeal"], "query_name": None, "slots": [], "required": True}]}, "PlanError"),
    (TimeoutError("planner timed out"), "TimeoutError"),
])
def test_any_model_failure_falls_back_to_rules_with_the_reason(translated, store, answer, reason):
    plan = planner.plan_with_model(None, Q, MENU, client=FakeClient(answer))
    assert plan.origin == "rules" and plan.shape == "simple" and plan.legs[0].text == Q
    assert plan.fallback_reason and reason in plan.fallback_reason


def test_a_fallback_is_logged_but_never_reused_and_a_later_model_plan_replaces_it(translated, store):
    first = planner.plan_with_model(None, Q, MENU, client=FakeClient("garbage"))
    assert first.origin == "rules" and len(store) == 1 and next(iter(store.values()))["origin"] == "rules"
    second = planner.plan_with_model(None, Q, MENU, client=FakeClient(GOOD))
    assert second.origin == "model" and next(iter(store.values()))["origin"] == "model"
    third = planner.plan_with_model(None, Q, MENU, client=FakeClient(RuntimeError("stored plan must answer")))
    assert third.origin == "model"


def test_stored_plan_key_changes_with_question_menu_or_model(translated, store, monkeypatch):
    planner.plan_with_model(None, Q, MENU, client=FakeClient(GOOD))
    other_menu = {**MENU, "source_docs": {**MENU["source_docs"], "appeal": "Appeal case files — now with letters"}}
    planner.plan_with_model(None, Q, other_menu, client=FakeClient(GOOD))
    planner.plan_with_model(None, Q + " Please.", MENU, client=FakeClient(GOOD))
    monkeypatch.setattr(planner, "PLANNER_MODEL", "claude-haiku-9-9-20990101")
    planner.plan_with_model(None, Q, MENU, client=FakeClient(GOOD))
    assert len(store) == 4


def test_rules_configuration_never_touches_the_model(monkeypatch):
    monkeypatch.setattr(planner, "PLANNER", "rules")
    monkeypatch.setattr(planner, "menu", lambda conn: pytest.fail("menu must not be read on the rules path"))
    plan = planner.plan_for(None, Q, client=FakeClient(RuntimeError("no")))
    assert plan.origin == "rules" and plan.shape == "simple"


def test_the_reader_call_is_stored_enforced_and_falls_back(translated, store):
    """Two calls (2026-09-12): the reading is its own constrained call with
    its own store key; the regex reader answers on failure and is never
    stored as the model's."""
    from raglab import router
    q = "For FEHB High, does the member have to pick a PCP?"
    reading = planner.read_with_model(None, q, client=FakeClient({**ROUTE, "program": "FEHB", "options": ["high"]}))
    assert reading.origin == "model" and reading.program == "FEHB" and reading.options == ("high",)
    assert router.route(q, reading=reading).plan_codes == ("71-006",)
    again = planner.read_with_model(None, q, client=FakeClient(RuntimeError("must not be called")))
    assert again == reading, "the stored reading is reused"
    other = "Does FEHB Standard need a referral?"
    fallback = planner.read_with_model(None, other, client=FakeClient(RuntimeError("down")))
    assert fallback.origin == "rules"
    retry = planner.read_with_model(None, other, client=FakeClient({**ROUTE, "program": "FEHB", "options": ["standard"]}))
    assert retry.origin == "model", "a fallback is never reused as the model's reading"


def test_an_impossible_reading_falls_back_to_the_regex_reader(translated, store):
    reading = planner.read_with_model(None, Q, client=FakeClient({**ROUTE, "years": [1999]}))
    assert reading.origin == "rules"


def test_reading_and_planning_are_separate_calls(translated, store):
    plan = planner.plan_with_model(None, Q, MENU, client=FakeClient(GOOD))
    assert plan.origin == "model" and "route" not in plan.to_dict()


def test_reading_and_plan_are_asked_of_the_model_concurrently_and_finished_in_order(db, monkeypatch):
    """Both store lookups miss: the two network calls overlap; the reading is
    finished at once, the plan only after the route is known (a pending plan)."""
    import threading
    import time as _time

    from raglab import planner, router

    monkeypatch.setattr(planner, "PLANNER", "model")
    monkeypatch.setattr(planner, "PLAN_CACHE", False)
    monkeypatch.setattr(planner, "_read_prepare", lambda conn, q: (("r",), q, None))
    monkeypatch.setattr(planner, "_plan_prepare", lambda conn, q, a: (("p",), q, None))
    active, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()

    def slow(result):
        def call(*a, **k):
            with lock:
                active["n"] += 1
                peak["n"] = max(peak["n"], active["n"])
            _time.sleep(0.15)
            with lock:
                active["n"] -= 1
            return result
        return call

    reading_raw = {"scope": "in_scope", "program": None, "options": [], "years": [2026], "change": False, "as_of": None}
    plan_raw = {"shape": "simple", "legs": [{"name": "brochure", "kind": "doc_probe", "text": "deductible", "sources": ["brochure"]}]}
    monkeypatch.setattr(planner, "_call_reader", slow(reading_raw))
    monkeypatch.setattr(planner, "_call_model", slow(plan_raw))
    monkeypatch.setattr(planner, "_validate_reading", lambda raw: None)
    monkeypatch.setattr(planner, "_reading_from_dict", lambda raw: router.Reading(scope="in_scope", origin="model"))
    available = {"sources": ("brochure",), "named_queries": ()}
    t0 = _time.perf_counter()
    reading, pending = planner.read_and_plan(db, "What is the deductible?", available, want_plan=True)
    assert _time.perf_counter() - t0 < 0.28, "the two calls ran one after the other"
    assert peak["n"] == 2 and reading.scope == "in_scope"
    assert isinstance(pending, planner._PendingPlan)
    plan = pending.finish(db, "What is the deductible?", available)
    assert plan.origin == "model" and tuple(plan.legs[0].sources) == ("brochure",) and plan.fallback_reason is None


def test_read_and_plan_asks_for_the_reading_only_when_no_plan_is_wanted(db, monkeypatch):
    from raglab import planner, router

    monkeypatch.setattr(planner, "PLANNER", "model")
    monkeypatch.setattr(planner, "PLAN_CACHE", False)
    monkeypatch.setattr(planner, "_read_prepare", lambda conn, q: (("r",), q, None))
    calls = []
    monkeypatch.setattr(planner, "_call_reader", lambda *a, **k: calls.append("read") or {"scope": "in_scope"})
    monkeypatch.setattr(planner, "_call_model", lambda *a, **k: calls.append("plan") or {})
    monkeypatch.setattr(planner, "_validate_reading", lambda raw: None)
    monkeypatch.setattr(planner, "_reading_from_dict", lambda raw: router.Reading(scope="in_scope", origin="model"))
    reading, pending = planner.read_and_plan(db, "q", {"sources": (), "named_queries": ()}, want_plan=False)
    assert calls == ["read"] and pending is None


def test_a_failed_plan_call_finishes_as_the_rules_plan_with_the_reason(db, monkeypatch):
    from raglab import planner

    monkeypatch.setattr(planner, "PLAN_CACHE", False)
    pending = planner._PendingPlan(("p",), "q", TimeoutError("planner timed out"), 1.0)
    plan = pending.finish(db, "What is the deductible?", {"sources": ("brochure",), "named_queries": ()})
    assert plan.origin == "rules" and plan.fallback_reason.startswith("TimeoutError")


def test_both_model_calls_are_deterministic_and_keyed_by_the_sampling(monkeypatch):
    """temperature 0 on the planner and the reader; the store key carries it,
    so plans sampled the old way never come back."""
    from raglab import planner

    seen = []

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                seen.append(kw)
                class _R:
                    content = [type("B", (), {"type": "text", "text": '{"shape": "simple", "legs": []}'})()]
                return _R()

    planner._call_model("q", {"sources": (), "named_queries": (), "source_docs": {}}, client=_Client())
    planner._call_reader("q", client=_Client())
    assert [kw["extra_body"]["temperature"] for kw in seen] == [0.0, 0.0]  # SDK 1.0 has no temperature parameter; the body field works
    assert planner.plan_key_model().endswith("@t0") and planner.plan_key_model().startswith(planner.PLANNER_MODEL)
