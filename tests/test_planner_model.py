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
