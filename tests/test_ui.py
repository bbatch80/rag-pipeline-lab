"""The surfaces render from the payload alone (Phase 5 decision 3): the
render contract in its fixed order for every status, from a fixture
payload with no database connected; then the pages over the Phase 4
identity — the portal shows the granted tiles, Ask runs the two-window
flagship. The page returns the payload and nothing else."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from raglab import context_services, identity, ui, webapp
from test_governance import _seed_tiers
from test_identical_question_control import _exact_scan, _fake_models
from test_webapp import PASSWORD, _Lease, _NoCommit

pytestmark = pytest.mark.readonly

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payload_ok.json").read_text())


def _order(html: str, *markers: str) -> list[int]:
    positions = [html.find(m) for m in markers]
    assert all(p >= 0 for p in positions), [m for m, p in zip(markers, positions) if p < 0]
    return positions


# ---------------------------------------------------------------- no database

def test_ok_renders_evidence_then_plan_line_then_footer_and_no_banner():
    html = ui.render_payload(FIXTURE)
    assert "banner" not in html
    evidence, plan, foot = _order(html, 'class="evidence"', 'class="planline"', 'class="payloadfoot"')
    assert evidence < plan < foot
    assert "draft" not in html.lower()  # the page returns the payload and nothing else
    assert FIXTURE["chunks"][0]["source"]["title"] in html and FIXTURE["payload_id"] in html
    assert "chip-acl" in html and "public" in html
    assert "1234 ms" in html and "(rerank 900)" in html
    # the warehouse leg: masked column labeled, masked value shown as masked, not as absent
    assert "masked: TOTAL_COST" in html and '<td class="masked"><span class="null">masked</span>' in html
    assert "M767394984" in html
    # the plan line names every leg with its outcome
    assert "Brochure search for 2026 High Option deductible (found)" in html
    assert "member_claims_summary (found)" in html and "plan by model" in html


def test_insufficient_evidence_banner_first():
    payload = {**FIXTURE, "status": "insufficient_evidence", "missing": ["member_claims_summary"], "warehouse_results": [],
               "unresolved_identifiers": ["M999900004"]}
    html = ui.render_payload(payload)
    banner, evidence = _order(html, "banner-insufficient", 'class="evidence"')
    assert banner < evidence
    assert "missing: member_claims_summary" in html and "Unresolved identifiers: M999900004" in html
    assert "member_claims_summary (nothing)" in html


def test_out_of_scope_shows_the_boundary_sentence_and_no_evidence():
    payload = {**FIXTURE, "status": "out_of_scope", "chunks": [], "warehouse_results": [], "plan": None,
               "boundary_response": "This platform answers questions about GEHA plans only."}
    html = ui.render_payload(payload)
    assert "banner-scope" in html and "GEHA plans only" in html
    assert 'class="evidence"' not in html and 'class="planline"' not in html
    assert 'class="payloadfoot"' in html  # the payload id is always shown


def test_single_probe_payload_without_a_plan_renders_the_route():
    payload = {**FIXTURE, "plan": None, "sub_results": [], "warehouse_results": []}
    html = ui.render_payload(payload)
    assert "looked in: 2026 · plans 71-006, 71-021" in html


def test_coverage_note_when_a_plan_is_missing_or_not_offered():
    payload = {**FIXTURE, "coverage": {**FIXTURE["coverage"], "missing_from_corpus": ["71-022"], "not_offered": []}}
    assert "missing from the corpus: 71-022" in ui.render_payload(payload)


def test_leg_outcomes_pair_legs_with_their_results():
    assert ui.leg_outcomes(FIXTURE) == [True, True]
    assert ui.leg_outcomes({**FIXTURE, "warehouse_results": [{"leg": "member_claims_summary", "status": "not_executed", "row_count": 0}]}) == [True, False]
    assert ui.leg_outcomes({"plan": None}) == []


# ---------------------------------------------------------------- the pages

@pytest.fixture
def app(db, monkeypatch):
    identity.seed(db, password=PASSWORD)
    _seed_tiers(db, per_tier=1, embed=True)
    _fake_models(monkeypatch)
    _exact_scan(db)
    # Ask composes through the planner; the pages' test stands the single
    # probe in for it so no model is called — the funnel, RLS, and disclosure are real
    monkeypatch.setattr(context_services, "compose",
                        lambda conn, ident, question, member_id=None, module=None, source="web", warehouse=None:
                        context_services.search(conn, ident, question, member_id=member_id, source=source))
    return webapp.create_app(connect=lambda: _Lease(_NoCommit(db)), warehouse=context_services.Warehouse(connect=None))


def _session(app, username):
    client = TestClient(app)
    resp = client.post("/ui/login", data={"username": username, "password": PASSWORD}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    return client


def test_portal_shows_the_granted_tiles_and_nothing_else(app):
    client = _session(app, "rep.dana")
    html = client.get("/").text
    assert 'href="/ask"' in html and 'href="/agent_assist"' in html
    assert 'href="/appeals_workbench"' not in html and 'href="/console"' not in html
    assert "Dana Okafor" in html and "call_center" in html
    assert TestClient(app).get("/", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/ask").status_code == 200
    bad = TestClient(app).post("/ui/login", data={"username": "rep.dana", "password": "nope"})
    assert bad.status_code == 401 and "wrong password" in bad.text


@pytest.mark.clean_corpus
def test_two_window_flagship_in_ask(app, db):
    """Two sessions, the same question in Ask: two payload ids, and each
    page shows exactly the documents that role is entitled to."""
    rep, cm = _session(app, "rep.dana"), _session(app, "cm.priya")
    rep_html = rep.post("/ui/ask", data={"question": "what do the secret facts say"}).text
    db.execute("RESET ROLE")
    cm_html = cm.post("/ui/ask", data={"question": "what do the secret facts say"}).text
    db.execute("RESET ROLE")
    ids = [re.search(r"payload <code>([0-9a-f-]{36})</code>", h).group(1) for h in (rep_html, cm_html)]
    assert ids[0] != ids[1]
    # the rep's tiers are public + employee; the care manager's clinical tier is
    # member-scoped, so with no member on the screen she sees public only —
    # and never the employee material the rep sees
    assert "doc-employee" in rep_html and "doc-care_team" not in rep_html
    assert "doc-employee" not in cm_html and "doc-member_services" not in cm_html
    assert "doc-public" in rep_html and "doc-public" in cm_html
    assert rep.post("/ui/draft", data={"payload_id": ids[0]}).status_code == 404  # no generated answer, by ruling
    assert rep.post("/ui/logout", follow_redirects=False).headers["location"] == "/login"
    assert rep.get("/ask", follow_redirects=False).status_code == 401
