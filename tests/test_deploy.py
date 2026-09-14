"""The deploy pieces that can be checked without a host: the smoke test's
verdicts against a fake site, the compose file's shape, and the app image's
recipe carrying what the serve path needs and nothing it does not."""

import re
from pathlib import Path

import httpx
import pytest

from raglab import smoke

ROOT = Path(__file__).resolve().parents[1]


def _site(*, login=200, status_ok=True, payload_status="ok", chunks=3):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/login" and request.method == "GET":
            return httpx.Response(200, text="<form>")
        if path == "/login":
            return httpx.Response(login, json={"surfaces": ["ask", "console"]} if login == 200 else {"detail": "no"})
        if path == "/status":
            return httpx.Response(200, json={"ok": status_ok, "counts": {"documents": 10, "chunks": 20},
                                             "problems": [] if status_ok else ["quarantined: x"]})
        if path == "/query":
            return httpx.Response(200, json={"payload_id": "p-1", "status": payload_status,
                                             "chunks": [{}] * chunks, "timings": {"total": 123.0}})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def _run(transport, **kw):
    real = httpx.Client

    class _Client(real):
        def __init__(self, *a, **k):
            k["transport"] = transport
            super().__init__(*a, **k)

    httpx.Client = _Client
    try:
        return smoke.run("https://example.test", password="pw", **kw)
    finally:
        httpx.Client = real


def test_smoke_passes_on_a_served_payload():
    out = _run(_site())
    assert out["health"] == {"ok": True, "documents": 10, "chunks": 20, "problems": []}
    assert out["payload"] == {"id": "p-1", "status": "ok", "chunks": 3, "total_ms": 123.0}


@pytest.mark.parametrize("site, message", [
    (_site(login=401), "login as admin"),
    (_site(status_ok=False), "/status reports problems"),
    (_site(payload_status="insufficient_evidence"), "did not return evidence"),
    (_site(chunks=0), "did not return evidence"),
])
def test_smoke_fails_closed(site, message):
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _run(site)


def test_smoke_needs_a_real_golden_question():
    with pytest.raises(RuntimeError, match="no golden question"):
        _run(_site(), question_id="nope-99")


def test_compose_declares_the_three_services_and_no_secret_defaults():
    text = (ROOT / "deploy" / "compose.yml").read_text()
    for service in ("db:", "app:", "caddy:"):
        assert service in text
    # secrets are required variables, never defaulted in the file
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RAGLAB_DEMO_PASSWORD", "RAGLAB_SESSION_SECRET"):
        assert re.search(rf"\${{{var}:\?", text), var
    assert "docker-entrypoint-initdb.d" in text and "/snapshot" in text
    example = (ROOT / "deploy" / ".env.example").read_text()
    for var in ("SITE_ADDRESS", "TAG", "RAGLAB_DEMO_PASSWORD", "RAGLAB_SESSION_SECRET"):
        assert f"{var}=" in example


def test_app_image_recipe_is_serve_only_with_weights_baked_in():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "--no-default-groups --group rerank" in dockerfile          # no parsers, no de-id models
    assert "snapshot_download" in dockerfile and "RERANKER_REVISION=" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile                             # never a runtime download
    assert "eval/golden.jsonl eval/baseline.json" in dockerfile         # the Console's numbers
    ignore = (ROOT / ".dockerignore").read_text().split()
    for path in ("data", "plans", ".env", "tests", "deploy/snapshot"):
        assert path in ignore, path


def test_db_init_creates_the_persona_roles_the_snapshot_expects():
    roles = (ROOT / "deploy" / "db-init" / "01-roles.sql").read_text()
    for role in ("persona_public", "persona_employee", "persona_care_team", "persona_member_services", "persona_appeals"):
        assert f"CREATE ROLE {role} NOLOGIN" in roles
    assert "GRANT persona_employee TO persona_member_services" in roles
    restore = (ROOT / "deploy" / "db-init" / "02-restore.sh").read_text()
    assert "pg_restore" in restore and "--no-owner" in restore and "/snapshot/raglab.dump" in restore
