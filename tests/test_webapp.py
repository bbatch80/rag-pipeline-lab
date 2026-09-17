"""The web doorway (Phase 4 PR1): a login session carries the identity, the
identity decides which surfaces open, and the platform maps a surface to
the planner module for the session's job. Nothing here touches retrieval;
the identity tables are seeded inside the rolled-back fixture transaction,
so the tests run on CI's fresh database."""

import pytest
from fastapi.testclient import TestClient

from raglab import identity, webapp

pytestmark = pytest.mark.readonly

PASSWORD = "pw"


class _Lease:
    """`with connect() as conn` over the fixture connection: hands out the
    same connection and neither commits nor closes it on exit."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


@pytest.fixture
def client(db):
    identity.seed(db, password=PASSWORD)
    app = webapp.create_app(connect=lambda: _Lease(db))
    with TestClient(app) as client:
        yield client


def _login(client, username, password=PASSWORD):
    return client.post("/login", json={"username": username, "password": password})


def test_no_session_is_401_and_logout_is_idempotent(client):
    assert client.get("/me").status_code == 401
    assert client.post("/logout").json() == {"logged_in": False}


def test_wrong_password_and_unknown_user_are_refused_alike(client):
    assert _login(client, "member_services", "nope").status_code == 401
    assert _login(client, "nobody").status_code == 401
    assert client.get("/me").status_code == 401


@pytest.mark.parametrize("username", sorted(identity.SEED_USERS))
def test_me_reflects_the_seeded_matrix(client, username):
    """/me is what the frontend renders from: persona, warehouse role, and
    the surfaces of the user's group — read from the tables, not the request."""
    _display, group = identity.SEED_USERS[username]
    persona, role, surfaces, _desc = identity.GROUPS[group]
    resp = _login(client, username)
    assert resp.status_code == 200, resp.text
    me = client.get("/me").json()
    assert me["username"] == username and me["group"] == group
    assert me["persona"] == persona and me["warehouse_role"] == role
    assert me["surfaces"] == sorted(surfaces)
    for surface in surfaces:
        assert client.get(f"/surfaces/{surface}").status_code == 200


def test_an_ungranted_surface_is_403_not_hidden(client):
    _login(client, "member_services")
    assert client.get("/surfaces/appeals_workbench").status_code == 403
    assert client.get("/surfaces/console").status_code == 403
    assert client.get("/surfaces/no_such_surface").status_code == 404
    _login(client, "appeals")  # a new login replaces the session
    assert client.get("/surfaces/appeals_workbench").status_code == 200
    assert client.get("/surfaces/agent_assist").status_code == 403


def test_the_same_screen_composes_under_the_job_s_module(client):
    """The rep and the care manager both open Agent Assist; the platform
    composes the rep under the agent_assist menu and the care manager under
    care_management — the page names the surface, the server picks the menu."""
    _login(client, "member_services")
    assert client.get("/surfaces/agent_assist").json() == {"surface": "agent_assist", "module": "agent_assist"}
    _login(client, "care_team")
    assert client.get("/surfaces/agent_assist").json() == {"surface": "agent_assist", "module": "care_management"}
    _login(client, "admin")
    assert client.get("/surfaces/console").json() == {"surface": "console", "module": None}
    assert client.get("/me").json()["modules"]["agent_assist"] == "*"  # the admin sees everything (ruling 2026-09-14)


def test_module_for_covers_every_surface_and_group():
    from raglab import planner

    for group in identity.GROUPS:
        for surface in webapp.SURFACES:
            module = webapp.module_for(surface, group)
            assert module is None or module == webapp.UNSCOPED or module in planner.MODULES, (surface, group, module)
    with pytest.raises(ValueError):
        webapp.module_for("benefits_lookup", "public")


def test_a_session_whose_user_vanished_is_logged_out(client, db):
    _login(client, "benefits")
    db.execute("DELETE FROM user_groups WHERE user_id = (SELECT id FROM users WHERE username = 'benefits')")
    db.execute("DELETE FROM users WHERE username = 'benefits'")
    assert client.get("/me").status_code == 401
    assert client.get("/me").status_code == 401  # the cookie was cleared, not merely refused once


# ---------------------------------------------------------------------------
# PR2: the three context services over HTTP — the same code as CLI and MCP.

from raglab import context_services, planner, snowlane  # noqa: E402
from test_governance import VISIBILITY, _seed_tiers  # noqa: E402  (tests/ is on sys.path)
from test_identical_question_control import _exact_scan, _fake_models, _tags  # noqa: E402

TIER_MAP = {role.removeprefix("persona_"): set(tags) for role, tags in VISIBILITY.items()}
MEMBER_SCOPED = {"care_team", "member_services", "appeals"}


class _NoCommit:
    """run_query commits its disclosure inside; the fixture must not."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, *a, **k):
        return self._conn.execute(*a, **k)

    def commit(self):
        pass

    def transaction(self):
        return self._conn.transaction()


class _FakeCursor:
    """Enough of a Snowflake cursor for run_named_query: the role, the tag,
    one result row shaped like member_claims_summary / cost_by_condition."""

    def __init__(self, role):
        self.role = role
        self.executed = []
        self.description = None

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "CURRENT_ROLE" in sql:
            self._rows = [(self.role,)]
            self.description = [("CURRENT_ROLE()",)]
        elif sql.startswith("ALTER SESSION"):
            self._rows = []
        else:
            self.description = [("MEMBER_ID",), ("LAST_NAME",), ("CLAIM_LINES",), ("TOTAL_COST",)]
            self._rows = [("M767394984", "Ng", 35, 1234.5)]
        return self

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows


class _FakeWarehouseConn:
    def __init__(self, role):
        self.role = role
        self.cursors = []

    def cursor(self):
        cur = _FakeCursor(self.role)
        self.cursors.append(cur)
        return cur

    def close(self):
        pass


@pytest.fixture
def client_with_corpus(db, monkeypatch):
    """A seeded tier corpus, fake embedding + reranker, exact scan: the real
    funnel (router, hybrid search, RLS, disclosure) minus the models."""
    identity.seed(db, password=PASSWORD)
    _seed_tiers(db, per_tier=1, embed=True)
    _fake_models(monkeypatch)
    _exact_scan(db)
    opened = {}
    warehouse = context_services.Warehouse(connect=lambda role: opened.setdefault(role, _FakeWarehouseConn(role)))
    app = webapp.create_app(connect=lambda: _Lease(_NoCommit(db)), warehouse=warehouse)
    with TestClient(app) as client:
        client.opened_warehouse = opened
        yield client


def _search(client, db, query="what do the secret facts say"):
    resp = client.post("/search", json={"query": query})
    db.execute("RESET ROLE")  # run_query's SET LOCAL ROLE normally ends with its commit; the proxy never commits
    return resp


@pytest.mark.clean_corpus  # the seeded tier documents must be the whole corpus
def test_same_question_two_users_two_payloads(client_with_corpus, db):
    """The durable Phase 4 test: identical question over HTTP as a rep and as
    a care manager → two payload ids and the content each tier is entitled to."""
    client = client_with_corpus
    _login(client, "member_services")
    rep = _search(client, db).json()
    _login(client, "care_team")
    cm = _search(client, db).json()
    assert rep["payload_id"] != cm["payload_id"]
    # person records (the rep's call notes, the care manager's clinical notes)
    # are searched only inside a member context; without one each sees the
    # tiers of documents about no one
    assert _tags(rep) == TIER_MAP["member_services"] - MEMBER_SCOPED == {"public", "employee"}
    assert _tags(cm) == TIER_MAP["care_team"] - MEMBER_SCOPED == {"public"}
    assert "care_team" not in _tags(rep) and "employee" not in _tags(cm)


@pytest.mark.clean_corpus  # the seeded tier documents must be the whole corpus
def test_identical_question_control_through_the_api(client_with_corpus, db):
    """Every seeded user, one question: exactly the tier map for the user's
    document persona (person records need a member context, so minus those)."""
    client = client_with_corpus
    for username, (_display, group) in identity.SEED_USERS.items():
        persona = identity.GROUPS[group][0]
        _login(client, username)
        got = _tags(_search(client, db).json())
        if persona == "admin":
            assert got == set().union(*TIER_MAP.values()) - MEMBER_SCOPED, username  # owner sees every tier
        else:
            assert got == TIER_MAP[persona] - MEMBER_SCOPED, username


def test_a_body_naming_an_identity_or_a_module_is_400_not_ignored(client_with_corpus):
    client = client_with_corpus
    _login(client, "member_services")
    for extra in ({"persona": "admin"}, {"role": "CLAIMS_EXAMINER"}, {"user_id": 1}, {"module": "appeals_workbench"}):
        resp = client.post("/query", json={"question": "q", "surface": "ask", **extra})
        assert resp.status_code == 400, extra
        assert list(extra)[0] in str(resp.json()["detail"])
    assert client.post("/search", json={"query": "q", "persona": "admin"}).status_code == 400
    assert client.post("/member-data", json={"query_name": "x", "surface": "ask", "role": "ACTUARY"}).status_code == 400
    assert client.post("/query", json={"question": "q", "surface": "ask"}).status_code == 200  # the same body without it


def test_query_composes_under_the_surface_s_module_as_the_session_identity(client_with_corpus, monkeypatch):
    seen = {}

    def fake_compose(conn, question, caller, member_id=None, plan=None, source="interactive", sf_connect=None, module=None, case_id=None):
        seen.update(question=question, caller=caller, member_id=member_id, case_id=case_id, source=source, module=module)
        sf_connect("CARE_MANAGER").close()  # a leg closes what it is handed; the shared session survives
        return {"status": "ok", "payload_id": "p"}

    monkeypatch.setattr(planner, "compose", fake_compose)
    client = client_with_corpus
    _login(client, "care_team")
    resp = client.post("/query", json={"question": "recent claims for this member", "surface": "agent_assist",
                                       "member_id": "M767394984"})
    assert resp.status_code == 200, resp.text
    assert seen["module"] == "care_management" and seen["source"] == "web" and seen["member_id"] == "M767394984"
    resp = client.post("/query", json={"question": "describe this appeal", "surface": "agent_assist", "case_id": "APL-0035378"})
    assert resp.status_code == 200 and seen["case_id"] == "APL-0035378" and seen["member_id"] is None
    assert seen["caller"].persona == "care_team" and seen["caller"].warehouse_role == "CARE_MANAGER"
    assert client.opened_warehouse["CARE_MANAGER"].cursors == []  # opened once, never closed by the leg
    assert client.post("/query", json={"question": "q", "surface": "appeals_workbench"}).status_code == 403
    assert client.post("/query", json={"question": "q", "surface": "console"}).status_code == 403
    _login(client, "admin")
    assert client.post("/query", json={"question": "q", "surface": "console"}).status_code == 400  # composes nothing
    client.post("/query", json={"question": "q", "surface": "agent_assist", "member_id": "M767394984"})
    assert seen["module"] is None  # the admin composes over the whole menu


def test_member_data_runs_the_surface_s_menu_as_the_warehouse_role(client_with_corpus):
    client = client_with_corpus
    _login(client, "analyst")
    resp = client.post("/member-data", json={"query_name": "cost_by_condition", "surface": "analyst_view",
                                             "params": {"description_like": "%asthma%"}})
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["status"] == "ok" and out["row_count"] == 1
    assert out["masked_columns"] == sorted(snowlane.MASKED_FOR_ROLE["ACTUARY"] & {"MEMBER_ID", "LAST_NAME", "CLAIM_LINES", "TOTAL_COST"})
    assert client.opened_warehouse["ACTUARY"].role == "ACTUARY" and "CARE_MANAGER" not in client.opened_warehouse
    tag = [p for sql, p in client.opened_warehouse["ACTUARY"].cursors[0].executed if sql.startswith("ALTER SESSION")][0]
    assert tag == ("raglab:cost_by_condition",)
    # a query that is not on this screen's menu is refused with the menu, before any warehouse call
    resp = client.post("/member-data", json={"query_name": "member_claims_summary", "surface": "analyst_view"})
    assert resp.status_code == 400 and "menu" in resp.json()["detail"]
    assert len(client.opened_warehouse["ACTUARY"].cursors) == 1
    # the rep's Agent Assist menu carries the member queries; the Analyst View is not hers
    _login(client, "member_services")
    assert client.post("/member-data", json={"query_name": "cost_by_condition", "surface": "analyst_view"}).status_code == 403
    assert client.post("/member-data", json={"query_name": "member_claims_summary", "surface": "agent_assist",
                                             "params": {"member_id": "M767394984"}}).json()["masked_columns"] == []


def test_an_identity_without_a_warehouse_role_gets_not_authorized():
    ident = identity.Identity(0, "benefits", "Sam", "benefits", "employee", None, ("ask",))
    out = context_services.member_data(ident, "member_claims_summary", {}, warehouse=context_services.Warehouse(connect=None))
    assert out["status"] == "not_authorized"


# ---------------------------------------------------------------------------
# PR3: Console reads — admin only; the disclosure log in both directions.

def test_console_reads_are_admin_only(client):
    for username in ("member_services", "care_team", "analyst", "appeals", "benefits"):
        _login(client, username)
        assert client.get("/status").status_code == 403, username
        assert client.get("/audit").status_code == 403, username
        assert client.get("/payload/00000000-0000-0000-0000-000000000000").status_code == 403, username
    client.post("/logout")
    assert client.get("/status").status_code == 401


def test_payload_schema_is_served_beside_the_payload_page(client):
    schema = client.get("/payload.schema.json")
    assert schema.status_code == 200 and schema.json()["title"] == "raglab context payload"
    assert client.get("/handoff").status_code == 404


@pytest.mark.clean_corpus
def test_a_payload_is_reproduced_exactly_from_the_disclosure_log(client_with_corpus, db):
    """A rep's search writes one disclosure row; the admin reads back the
    same payload, who asked, and what it was built from."""
    client = client_with_corpus
    _login(client, "member_services")
    delivered = _search(client, db, "what do the secret facts say").json()
    _login(client, "admin")
    resp = client.get(f"/payload/{delivered['payload_id']}")
    assert resp.status_code == 200, resp.text
    rec = resp.json()
    # stage timings are stamped after the disclosure row is written (they time
    # the disclose step itself), so the record holds everything but them
    assert rec["payload"] == {k: v for k, v in delivered.items() if k != "timings"}
    assert rec["username"] == "member_services" and rec["persona"] == "member_services" and rec["source"] == "web"
    assert rec["payload_status"] == delivered["status"] and set(rec["acl_basis"]) == {"public", "employee"}
    assert len(rec["chunk_ids"]) == len(delivered["chunks"]) == len(rec["content_hashes"])
    assert client.get("/payload/00000000-0000-0000-0000-000000000000").status_code == 404


@pytest.mark.clean_corpus
def test_audit_reads_both_directions(client_with_corpus, db):
    client = client_with_corpus
    _login(client, "member_services")
    _search(client, db)
    _login(client, "care_team")
    _search(client, db)
    _login(client, "admin")
    everything = client.get("/audit").json()
    assert everything["totals"]["disclosures"] >= 2 and everything["totals"]["users"] >= 2
    who = client.get("/audit", params={"username": "care_team"}).json()
    assert who["rows"] and {r["username"] for r in who["rows"]} == {"care_team"} and {r["persona"] for r in who["rows"]} == {"care_team"}
    by_persona = client.get("/audit", params={"persona": "member_services"}).json()
    assert by_persona["rows"] and {r["persona"] for r in by_persona["rows"]} == {"member_services"}
    which = client.get("/audit", params={"document": "doc-employee"}).json()  # lineage: which payloads used this document
    assert which["rows"] and all(any("doc-employee" in t for t in r["doc_titles"]) for r in which["rows"])
    assert {r["username"] for r in which["rows"]} == {"member_services"}  # the care manager cannot have used an employee document
    assert client.get("/audit", params={"limit": 1}).json()["rows"].__len__() == 1


def test_status_is_the_health_snapshot_as_data(client, monkeypatch):
    from raglab import pipeline

    monkeypatch.setitem(pipeline.DISCLOSURE_FAILURES, "count", 0)  # process memory; another test may have tripped it
    _login(client, "admin")
    out = client.get("/status").json()
    assert {"postgres", "pgvector", "counts", "hnsw_index", "disclosure_failures", "baseline", "problems", "ok"} <= set(out)
    assert set(out["counts"]) >= {"documents", "chunks", "quarantine", "disclosure_log"}
    assert out["disclosure_failures"] == 0 and out["latency_budget_p95_ms"] == 1000
    if out["baseline"]:
        assert out["baseline"]["item_pass"] and out["baseline"]["run_id"]
    monkeypatch.setitem(pipeline.DISCLOSURE_FAILURES, "count", 2)
    out = client.get("/status").json()
    assert out["disclosure_failures"] == 2 and not out["ok"] and any("withheld" in p for p in out["problems"])
