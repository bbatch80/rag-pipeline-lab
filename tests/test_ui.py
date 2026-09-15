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


class _Answerer:
    name = "fake-answerer"

    def generate(self, payload):
        return f"**Yes.** Drawn from {len(payload['chunks'])} chunks.\n\nSee <71-006> page 9."


def test_answer_renders_first_from_the_payload_and_never_without_evidence(monkeypatch):
    """User ruling 2026-09-14: the surface shows the model's answer, the
    evidence beneath it — generated only when the platform served evidence."""
    monkeypatch.setenv("RAGLAB_ANSWERS", "on")

    class _App:
        class state:
            generator = _Answerer

    answer = ui.answer_for(_App, FIXTURE)
    html = ui.render_answer(answer) + ui.render_payload(FIXTURE)
    a, evidence = _order(html, 'class="answer"', 'class="evidence"')
    assert a < evidence and "fake-answerer" in html
    assert "<strong>Yes.</strong>" in html and "&lt;71-006&gt;" in html  # bold allowed, markup escaped
    assert ui.answer_for(_App, {**FIXTURE, "status": "insufficient_evidence"}) is None
    assert ui.answer_for(_App, {**FIXTURE, "chunks": [], "warehouse_results": []}) is None
    monkeypatch.setenv("RAGLAB_ANSWERS", "off")
    assert ui.answer_for(_App, FIXTURE) is None

    class _Broken:
        name = "broken"

        def generate(self, payload):
            raise RuntimeError("credit balance is too low")

    monkeypatch.setenv("RAGLAB_ANSWERS", "on")
    _App.state.generator = _Broken
    out = ui.render_answer(ui.answer_for(_App, FIXTURE))
    assert "No answer could be generated" in out and "credit balance" in out
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


def test_a_real_out_of_scope_payload_renders_without_error():
    """The shape compose actually returns for out_of_scope (2026-09-14: the
    Medicare Part B question 500'd the page because the footer assumed a
    confidence field): no confidence, no plan, no router, no coverage."""
    payload = {"spec_version": "1.1.0", "query": "How long do I have after I retire to enroll in Medicare Part B?",
               "status": "out_of_scope", "boundary_response": "Medicare program facts are outside this corpus.",
               "retrieved_at": "2026-09-14T14:52:40+00:00", "chunks": [], "payload_id": "abc", "persona": "care_team",
               "member_context": None, "record_context": {}, "unresolved_identifiers": [], "plan": None,
               "sub_results": [], "warehouse_results": [], "missing": [], "timings": {"total": 12.0}}
    html = ui.render_payload(payload)
    assert "banner-scope" in html and "outside this corpus" in html and "payload <code>abc</code>" in html
    assert ui.answer_for(type("A", (), {"state": type("S", (), {"generator": None})}), payload) is None


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
                        lambda conn, ident, question, member_id=None, case_id=None, module=None, source="web", warehouse=None:
                        context_services.search(conn, ident, question, member_id=member_id, source=source))
    app = webapp.create_app(connect=lambda: _Lease(_NoCommit(db)), warehouse=context_services.Warehouse(connect=None))
    app.state.generator = _Answerer
    return app


def _session(app, username):
    client = TestClient(app)
    resp = client.post("/ui/login", data={"username": username, "password": PASSWORD}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    return client


def test_portal_shows_the_granted_tiles_and_nothing_else(app):
    client = _session(app, "member_services")
    html = client.get("/").text
    assert 'href="/ask"' in html and 'href="/agent_assist"' in html
    assert 'href="/appeals_workbench"' not in html and 'href="/console"' not in html
    assert "Member services rep" in html and "MEMBER_SERVICES_REP" in html
    assert TestClient(app).get("/", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/ask").status_code == 200
    bad = TestClient(app).post("/ui/login", data={"username": "member_services", "password": "nope"})
    assert bad.status_code == 401 and "wrong password" in bad.text


@pytest.mark.clean_corpus
def test_two_window_flagship_in_ask(app, db):
    """Two sessions, the same question in Ask: two payload ids, and each
    page shows exactly the documents that role is entitled to."""
    rep, cm = _session(app, "member_services"), _session(app, "care_team")
    rep_html = rep.post("/ui/ask", data={"question": "what do the secret facts say"}).text
    db.execute("RESET ROLE")
    cm_html = cm.post("/ui/ask", data={"question": "what do the secret facts say"}).text
    db.execute("RESET ROLE")
    ids = [re.search(r"payload <code>([0-9a-f-]{36})</code>", h).group(1) for h in (rep_html, cm_html)]
    assert ids[0] != ids[1]
    assert rep_html.index('class="answer"') < rep_html.index('class="evidence"')  # the answer first, from the evidence
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
        elif "FROM PATIENTS p" in sql:  # member_profile: who the member is + current enrollment
            self.description = [(c,) for c in ("MEMBER_ID", "MRN", "FIRST_NAME", "LAST_NAME", "BIRTHDATE", "GENDER", "CITY", "STATE", "ZIP",
                                               "LINE_OF_BUSINESS", "ENROLLMENT_YEAR", "PLAN_CODE", "PLAN_OPTION", "TIER", "ENROLLMENT_CODE")]
            self._rows = [(FAKE_MEMBER, "MRN0000001", "Test", "Member", "1970-01-01", "F", "Kansas City", "MO", "641",
                           "FEHB", 2026, "71-006", "High", "Self Only", "311")] if params["member_id"] == FAKE_MEMBER else []
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

    def compose(conn, ident, question, member_id=None, case_id=None, module=None, source="web", warehouse=None):
        seen.append({"question": question, "member_id": member_id, "case_id": case_id, "module": module, "persona": ident.persona})
        return context_services.search(conn, ident, question, member_id=member_id, source=source)

    monkeypatch.setattr(context_services, "compose", compose)
    app = webapp.create_app(connect=lambda: _Lease(_NoCommit(db)),
                            warehouse=context_services.Warehouse(connect=lambda role: _RecordWarehouse(role)))
    app.state.seen = seen
    app.state.generator = _Answerer
    return app


def test_agent_assist_loads_nothing_until_a_valid_known_member_is_typed(surfaces):
    rep = _session(surfaces, "member_services")
    page = rep.get("/agent_assist").text
    assert "Nothing is loaded until you open a member" in page and "headerrow" not in page and FAKE_MEMBER not in page
    bad = rep.post("/ui/agent_assist/open", data={"member_id": "M12345"}).text
    assert "not a valid member id" in bad and "headerrow" not in bad
    nobody = rep.post("/ui/agent_assist/open", data={"member_id": UNKNOWN_MEMBER}).text
    assert f"No member id {UNKNOWN_MEMBER} on record" in nobody and "headerrow" not in nobody
    opened = rep.post("/ui/agent_assist/open", data={"member_id": " m999900004 "}).text  # surface form → canonical
    assert f"Member {FAKE_MEMBER} is open" in opened and 'hx-post="/ui/agent_assist/ask"' in opened
    assert "headerrow" not in opened and "1970-01-01" not in opened and "71-006" not in opened  # nothing rendered until a question is asked
    assert f'name="member_id" value="{FAKE_MEMBER}"' in opened
    assert 'value=""' in opened  # the question box starts empty: no canned question


@pytest.mark.clean_corpus
def test_rep_s_agent_assist_shows_no_clinical_text_and_the_care_manager_s_does(surfaces, db):
    """The durable Phase 5 test: the same member, the same screen, the same
    question — the rep's page carries no clinical note text; the care
    manager's does, and carries none of the rep's call-note tier."""
    rep, cm = _session(surfaces, "member_services"), _session(surfaces, "care_team")
    rep_html = rep.post("/ui/agent_assist/ask", data={"question": "what do the secret facts say", "member_id": FAKE_MEMBER}).text
    db.execute("RESET ROLE")
    cm_html = cm.post("/ui/agent_assist/ask", data={"question": "what do the secret facts say", "member_id": FAKE_MEMBER}).text
    db.execute("RESET ROLE")
    assert "doc-care_team" not in rep_html and "doc-member_services" in rep_html and "doc-employee" in rep_html
    assert "doc-care_team" in cm_html and "doc-member_services" not in cm_html and "doc-employee" not in cm_html
    assert f"member {FAKE_MEMBER}" in rep_html  # the plan line names the member context
    modules = [s["module"] for s in surfaces.state.seen]
    assert modules == ["agent_assist", "care_management"]  # same screen, the job's module
    assert all(s["member_id"] == FAKE_MEMBER and s["case_id"] is None for s in surfaces.state.seen)


def test_workbench_opens_a_case_and_starts_the_question_with_it(surfaces):
    analyst = _session(surfaces, "appeals")
    assert "Nothing is loaded until you open a case" in analyst.get("/appeals_workbench").text
    assert "not a valid case id" in analyst.post("/ui/appeals_workbench/open", data={"case_id": "APL-000000"}).text
    assert "on record" in analyst.post("/ui/appeals_workbench/open", data={"case_id": identifiers.case_id(8)}).text
    opened = analyst.post("/ui/appeals_workbench/open", data={"case_id": KNOWN_CASE}).text
    assert f"Case {KNOWN_CASE} is open" in opened and "not medically necessary" not in opened  # validated and found; nothing rendered yet
    assert f'name="case_id" value="{KNOWN_CASE}"' in opened   # the open case is the context, as page state
    assert 'value=""' in opened and "Case APL" not in opened.split("askbox")[1][:200]  # no prefix: ask naturally
    assert "Evidence, never a determination" in opened
    analyst.post("/ui/appeals_workbench/ask", data={"question": "Describe this appeal.", "case_id": KNOWN_CASE})
    assert surfaces.state.seen[-1] == {"question": "Describe this appeal.", "member_id": None, "case_id": KNOWN_CASE,
                                       "module": "appeals_workbench", "persona": "appeals"}


def test_surfaces_are_granted_per_group(surfaces):
    rep = _session(surfaces, "member_services")
    assert rep.get("/appeals_workbench").status_code == 403
    assert rep.post("/ui/appeals_workbench/open", data={"case_id": KNOWN_CASE}).status_code == 403
    analyst = _session(surfaces, "appeals")
    assert analyst.get("/agent_assist").status_code == 403
    sam = _session(surfaces, "benefits")
    assert sam.get("/agent_assist").status_code == 403 and sam.get("/ask").status_code == 200


# ---------------------------------------------------------------------------
# PR3: Analyst View and the Console.

class _AggregateCursor(_RecordCursor):
    """Adds the analyst's aggregate to the record cursor."""

    def execute(self, sql, params=None):
        if "GROUP BY" in sql or "description_like" in sql:
            self.description = [(c,) for c in ("DESCRIPTION", "MEMBERS", "CLAIM_LINES", "AVG_COST", "TOTAL_COST")]
            self._rows = [("Asthma follow-up", 412, 1830, 96.4, 176412.0)] if params.get("description_like") else []
            return self
        return super().execute(sql, params)


class _AggregateWarehouse(_RecordWarehouse):
    def cursor(self):
        return _AggregateCursor(self.role)


@pytest.fixture
def analyst_app(db, monkeypatch):
    identity.seed(db, password=PASSWORD)
    return webapp.create_app(connect=lambda: _Lease(_NoCommit(db)),
                             warehouse=context_services.Warehouse(connect=lambda role: _AggregateWarehouse(role)))


def test_analyst_view_offers_the_menu_and_runs_as_the_warehouse_role(analyst_app):
    jo = _session(analyst_app, "analyst")
    page = jo.get("/analyst_view").text
    for q in ("cost_by_condition", "providers_by_specialty", "provider_network_status"):
        assert f'<option value="{q}">' in page
    assert "member_claims_summary" not in page  # not on this screen's menu
    params = jo.get("/ui/analyst_view/params", params={"query_name": "cost_by_condition"}).text
    assert 'name="description_like"' in params and "Aggregate across members" in params
    assert jo.get("/ui/analyst_view/params", params={"query_name": "member_claims_summary"}).status_code == 400
    out = jo.post("/ui/analyst_view/run", data={"query_name": "cost_by_condition", "description_like": "%asthma%"}).text
    assert "Asthma follow-up" in out and "176412.0" in out and "1 row" in out
    assert "chip-masked" not in out  # nothing identifying in an aggregate: nothing masked
    off_menu = jo.post("/ui/analyst_view/run", data={"query_name": "member_claims_summary", "member_id": FAKE_MEMBER})
    assert off_menu.status_code == 400
    assert _session(analyst_app, "member_services").get("/analyst_view").status_code == 403


@pytest.mark.clean_corpus
def test_console_health_audit_and_payload_page(surfaces, db, monkeypatch, tmp_path):
    from raglab import dashboard, pipeline

    monkeypatch.setitem(pipeline.DISCLOSURE_FAILURES, "count", 0)
    rep = _session(surfaces, "member_services")
    html = rep.post("/ui/ask", data={"question": "what do the secret facts say"}).text
    db.execute("RESET ROLE")
    pid = re.search(r"payload <code>([0-9a-f-]{36})</code>", html).group(1)
    assert rep.get("/console").status_code == 403 and rep.get(f"/console/payload/{pid}").status_code == 403

    admin = _session(surfaces, "admin")
    page = admin.get("/console").text
    assert "Platform Console" in page and "documents" in page and "gate baseline" in page and "/console/dashboard" in page
    rows = admin.get("/ui/console/audit", params={"username": "member_services"}).text
    assert "member_services" in rows and "member_services" in rows and pid[:8] in rows and "secret facts" in rows
    assert "No disclosures match" in admin.get("/ui/console/audit", params={"username": "nobody"}).text
    jump = admin.get("/ui/console/audit", params={"payload_id": pid}, follow_redirects=False)
    assert jump.status_code == 303 and jump.headers["location"] == f"/console/payload/{pid}"
    detail = admin.get(f"/console/payload/{pid}").text
    assert f"QUERY_TAG = {pid}" in detail and "member_services" in detail and 'class="evidence"' in detail and "doc-employee" in detail
    assert admin.get("/console/payload/00000000-0000-0000-0000-000000000000").status_code == 404
    # the scorecard: the dashboard file as the last full run wrote it, or a pointer when none exists
    monkeypatch.setattr(dashboard, "OUT_PATH", tmp_path / "none.html")
    assert "No dashboard yet" in admin.get("/console/dashboard").text
    (tmp_path / "none.html").write_text("<title>raglab — evaluation dashboard</title><p>scorecard</p>")
    assert "scorecard" in admin.get("/console/dashboard").text
