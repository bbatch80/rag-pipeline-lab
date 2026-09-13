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
    assert _login(client, "rep.dana", "nope").status_code == 401
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
    _login(client, "rep.dana")
    assert client.get("/surfaces/appeals_workbench").status_code == 403
    assert client.get("/surfaces/console").status_code == 403
    assert client.get("/surfaces/no_such_surface").status_code == 404
    _login(client, "appeals.lee")  # a new login replaces the session
    assert client.get("/surfaces/appeals_workbench").status_code == 200
    assert client.get("/surfaces/agent_assist").status_code == 403


def test_the_same_screen_composes_under_the_job_s_module(client):
    """The rep and the care manager both open Agent Assist; the platform
    composes the rep under the agent_assist menu and the care manager under
    care_management — the page names the surface, the server picks the menu."""
    _login(client, "rep.dana")
    assert client.get("/surfaces/agent_assist").json() == {"surface": "agent_assist", "module": "agent_assist"}
    _login(client, "cm.priya")
    assert client.get("/surfaces/agent_assist").json() == {"surface": "agent_assist", "module": "care_management"}
    _login(client, "admin")
    assert client.get("/surfaces/console").json() == {"surface": "console", "module": None}
    assert client.get("/me").json()["modules"]["agent_assist"] == "agent_assist"


def test_module_for_covers_every_surface_and_group():
    from raglab import planner

    for group in identity.GROUPS:
        for surface in webapp.SURFACES:
            module = webapp.module_for(surface, group)
            assert module is None or module in planner.MODULES, (surface, group, module)
    with pytest.raises(ValueError):
        webapp.module_for("benefits_lookup", "public")


def test_a_session_whose_user_vanished_is_logged_out(client, db):
    _login(client, "benefits.sam")
    db.execute("DELETE FROM user_groups WHERE user_id = (SELECT id FROM users WHERE username = 'benefits.sam')")
    db.execute("DELETE FROM users WHERE username = 'benefits.sam'")
    assert client.get("/me").status_code == 401
    assert client.get("/me").status_code == 401  # the cookie was cleared, not merely refused once
