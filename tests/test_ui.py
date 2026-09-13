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


# ---------------------------------------------------------------------------
# PR2: Agent Assist and the Appeals Workbench — nothing loads until a key
# is typed; the key is validated like the pipeline validates it; one header
# row confirms what is open; every panel is a question.

import psycopg  # noqa: E402

from raglab import identifiers  # noqa: E402
from test_identical_question_control import FAKE_MEMBER, FAKE_PATIENT  # noqa: E402

KNOWN_CASE = identifiers.case_id(7)          # APL-1215086, check digit valid
UNKNOWN_MEMBER = "M999900012"                # shape + check digit valid, nobody's


class _RecordCursor:
    """A warehouse cursor that answers the two header queries: an enrollment
    row for the fake member, the case row for the known case, nothing else."""

    def __init__(self, role):
        self.role = role
        self.description = None
        self._rows = []

    def execute(self, sql, params=None):
        if "CURRENT_ROLE" in sql:
            self._rows, self.description = [(self.role,)], [("CURRENT_ROLE()",)]
        elif sql.startswith("ALTER SESSION"):
            self._rows = []
        elif "FROM ENROLLMENT" in sql:
            self.description = [(c,) for c in ("MEMBER_ID", "YEAR", "LINE_OF_BUSINESS", "PLAN_CODE", "PLAN_OPTION", "TIER", "ENROLLMENT_CODE")]
            self._rows = [(FAKE_MEMBER, 2026, "FEHB", "71-006", "High", "Self Only", "311")] if params["member_id"] == FAKE_MEMBER else []
        elif "FROM APPEALS" in sql:
            self.description = [(c,) for c in ("CASE_ID", "MEMBER_ID", "CLAIM_ID", "CALL_ID", "FILED_DATE", "APPEAL_TYPE",
                                               "DENIAL_REASON", "DECISION", "DECIDED_DATE", "REVIEWER", "POLICY_ID")]
            self._rows = [(KNOWN_CASE, FAKE_MEMBER, "CLM-136378150", "CALL-1", "2025-03-04", "clinical",
                           "not medically necessary", "upheld", "2025-03-28", "R. Vale", "CP-0003")] if params["case_id"] == KNOWN_CASE else []
        else:
            raise AssertionError(f"unexpected warehouse query: {sql[:60]}")
        return self

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows


class _RecordWarehouse:
    def __init__(self, role):
        self.role = role

    def cursor(self):
        return _RecordCursor(self.role)

    def close(self):
        pass


@pytest.fixture
def surfaces(db, monkeypatch):
    """Seeded tiers scoped to one fake member, fake models, a fake
    warehouse answering the header queries; Ask-style composition stands
    in for the planner."""
    identity.seed(db, password=PASSWORD)
    _seed_tiers(db, per_tier=1, embed=True)
    _fake_models(monkeypatch)
    _exact_scan(db)
    try:
        with db.transaction():
            db.execute("INSERT INTO synthea.patients (id, member_id) VALUES (%s, %s)", (FAKE_PATIENT, FAKE_MEMBER))
    except (psycopg.errors.UndefinedTable, psycopg.errors.NotNullViolation) as exc:
        pytest.skip(f"no usable synthea.patients here: {exc.__class__.__name__}")
    db.execute("UPDATE documents SET member_key = %s WHERE title LIKE 'doc-%%'", (FAKE_PATIENT,))
    db.execute("UPDATE chunks SET member_key = %s WHERE document_id IN (SELECT id FROM documents WHERE title LIKE 'doc-%%')", (FAKE_PATIENT,))
    seen = []

    def compose(conn, ident, question, member_id=None, module=None, source="web", warehouse=None):
        seen.append({"question": question, "member_id": member_id, "module": module, "persona": ident.persona})
        return context_services.search(conn, ident, question, member_id=member_id, source=source)

    monkeypatch.setattr(context_services, "compose", compose)
    app = webapp.create_app(connect=lambda: _Lease(_NoCommit(db)),
                            warehouse=context_services.Warehouse(connect=lambda role: _RecordWarehouse(role)))
    app.state.seen = seen
    return app


def test_agent_assist_loads_nothing_until_a_valid_known_member_is_typed(surfaces):
    rep = _session(surfaces, "rep.dana")
    page = rep.get("/agent_assist").text
    assert "Nothing is loaded until you open a member" in page and "headerrow" not in page and FAKE_MEMBER not in page
    bad = rep.post("/ui/agent_assist/open", data={"member_id": "M12345"}).text
    assert "not a valid member id" in bad and "headerrow" not in bad
    nobody = rep.post("/ui/agent_assist/open", data={"member_id": UNKNOWN_MEMBER}).text
    assert f"No member id {UNKNOWN_MEMBER} on record" in nobody and "headerrow" not in nobody
    opened = rep.post("/ui/agent_assist/open", data={"member_id": " m999900004 "}).text  # surface form → canonical
    assert "headerrow" in opened and f"Member {FAKE_MEMBER} — enrollment" in opened and "71-006" in opened
    assert f'name="member_id" value="{FAKE_MEMBER}"' in opened and 'hx-post="/ui/agent_assist/ask"' in opened
    assert 'value=""' in opened  # the question box starts empty: no canned question


@pytest.mark.clean_corpus
def test_rep_s_agent_assist_shows_no_clinical_text_and_the_care_manager_s_does(surfaces, db):
    """The durable Phase 5 test: the same member, the same screen, the same
    question — the rep's page carries no clinical note text; the care
    manager's does, and carries none of the rep's call-note tier."""
    rep, cm = _session(surfaces, "rep.dana"), _session(surfaces, "cm.priya")
    rep_html = rep.post("/ui/agent_assist/ask", data={"question": "what do the secret facts say", "member_id": FAKE_MEMBER}).text
    db.execute("RESET ROLE")
    cm_html = cm.post("/ui/agent_assist/ask", data={"question": "what do the secret facts say", "member_id": FAKE_MEMBER}).text
    db.execute("RESET ROLE")
    assert "doc-care_team" not in rep_html and "doc-member_services" in rep_html and "doc-employee" in rep_html
    assert "doc-care_team" in cm_html and "doc-member_services" not in cm_html and "doc-employee" not in cm_html
    assert f"member {FAKE_MEMBER}" in rep_html  # the plan line names the member context
    modules = [s["module"] for s in surfaces.state.seen]
    assert modules == ["agent_assist", "care_management"]  # same screen, the job's module
    assert all(s["member_id"] == FAKE_MEMBER for s in surfaces.state.seen)


def test_workbench_opens_a_case_and_starts_the_question_with_it(surfaces):
    analyst = _session(surfaces, "appeals.lee")
    assert "Nothing is loaded until you open a case" in analyst.get("/appeals_workbench").text
    assert "not a valid case id" in analyst.post("/ui/appeals_workbench/open", data={"case_id": "APL-000000"}).text
    assert "on record" in analyst.post("/ui/appeals_workbench/open", data={"case_id": identifiers.case_id(8)}).text
    opened = analyst.post("/ui/appeals_workbench/open", data={"case_id": KNOWN_CASE}).text
    assert f"Case {KNOWN_CASE}" in opened and "not medically necessary" in opened and "CP-0003" in opened
    assert f'value="Case {KNOWN_CASE}: "' in opened  # the box starts with the case, visible and editable
    assert f'name="member_id" value="{FAKE_MEMBER}"' in opened  # the case's member, from the platform's row
    assert "Evidence, never a determination" in opened
    analyst.post("/ui/appeals_workbench/ask", data={"question": f"Case {KNOWN_CASE}: why was it upheld?", "member_id": FAKE_MEMBER})
    assert surfaces.state.seen[-1] == {"question": f"Case {KNOWN_CASE}: why was it upheld?", "member_id": FAKE_MEMBER,
                                       "module": "appeals_workbench", "persona": "appeals"}


def test_surfaces_are_granted_per_group(surfaces):
    rep = _session(surfaces, "rep.dana")
    assert rep.get("/appeals_workbench").status_code == 403
    assert rep.post("/ui/appeals_workbench/open", data={"case_id": KNOWN_CASE}).status_code == 403
    analyst = _session(surfaces, "appeals.lee")
    assert analyst.get("/agent_assist").status_code == 403
    sam = _session(surfaces, "benefits.sam")
    assert sam.get("/agent_assist").status_code == 403 and sam.get("/ask").status_code == 200
