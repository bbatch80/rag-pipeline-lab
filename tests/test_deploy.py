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


def test_release_workflow_builds_then_deploys_behind_the_production_gate_and_rolls_back():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert 'tags: ["v2.*"]' in text                                    # a tag is a release
    assert "name: production" in text and "environment:" in text       # the approval gate
    assert "needs: build" in text                                      # images first
    assert "deploy.sh up ${{ needs.build.outputs.version }}" in text
    assert "raglab smoke" in text                                      # the live check
    assert "deploy.sh rollback" in text and "if: failure() && steps.deploy.outcome == 'success'" in text
    for secret in ("DEPLOY_HOST", "DEPLOY_USER", "DEPLOY_SSH_KEY", "RAGLAB_DEMO_PASSWORD"):
        assert f"secrets.{secret}" in text, secret
    assert "vars.SITE_ADDRESS" in text


def test_deploy_script_parses_and_documents_its_verbs():
    import subprocess

    script = ROOT / "deploy" / "deploy.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    text = script.read_text()
    for verb in ('  up)', '  rollback)', '  status)'):
        assert verb in text
    assert ".previous-tag" in text and "--no-pull" in text


def test_deploy_script_up_and_rollback_switch_the_tag(tmp_path):
    """Exercise the tag bookkeeping with compose stubbed out: up records the
    previous tag and writes the new one; rollback restores it."""
    import shutil
    import subprocess

    root = tmp_path / "raglab"
    (root / "deploy").mkdir(parents=True)
    shutil.copy(ROOT / "deploy" / "deploy.sh", root / "deploy" / "deploy.sh")
    (root / "deploy" / "compose.yml").write_text("services: {}\n")
    (root / "deploy" / ".env").write_text("SITE_ADDRESS=localhost\nTAG=v2.0.0\n")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "docker").write_text("#!/bin/sh\necho docker \"$@\" >> \"$DOCKER_LOG\"\n")
    (stub / "docker").chmod(0o755)
    env = {"PATH": f"{stub}:/usr/bin:/bin", "DOCKER_LOG": str(tmp_path / "docker.log")}

    def run(*args):
        return subprocess.run(["bash", str(root / "deploy" / "deploy.sh"), *args], env=env, capture_output=True, text=True)

    assert run("up", "v2.0.1").returncode == 0
    assert "TAG=v2.0.1" in (root / "deploy" / ".env").read_text()
    assert (root / "deploy" / ".previous-tag").read_text().strip() == "v2.0.0"
    log = (tmp_path / "docker.log").read_text()
    assert "pull --quiet app db" in log and "up -d --remove-orphans" in log
    assert run("rollback").returncode == 0
    assert "TAG=v2.0.0" in (root / "deploy" / ".env").read_text()
    assert "SITE_ADDRESS=localhost" in (root / "deploy" / ".env").read_text()  # the rest of the file untouched
    (tmp_path / "docker.log").write_text("")
    assert run("up", "v2.0.2", "--no-pull").returncode == 0
    assert "pull" not in (tmp_path / "docker.log").read_text()
    assert run("nonsense").returncode == 2


def test_smoke_waits_for_a_site_that_comes_up_late(monkeypatch):
    """A fresh deploy: TLS errors, then 503, then 200 — the wait absorbs it."""
    monkeypatch.setattr(smoke.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login" and request.method == "GET":
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("tlsv1 alert internal error")
            if calls["n"] == 2:
                return httpx.Response(503)
            return httpx.Response(200, text="<form>")
        return _site().handler(request)  # the rest of the site as in the passing case

    out = _run(httpx.MockTransport(handler), wait=60)
    assert calls["n"] >= 3 and out["payload"]["status"] == "ok"


def test_smoke_gives_up_after_the_wait(monkeypatch):
    monkeypatch.setattr(smoke.time, "sleep", lambda s: None)
    clock = {"t": 0.0}
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 10) or clock["t"])
    transport = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(RuntimeError, match="login page not up after 30s: ConnectError"):
        _run(transport, wait=30)


def test_deploy_prunes_every_release_image_but_the_running_and_incoming_tags(tmp_path, monkeypatch):
    """Fifteen 4.4 GB release images filled the host and a deploy failed
    mid-pull (2026-09-15). Before pulling, `up` keeps only the tag being
    deployed and the one running (the rollback target); the registry holds
    the rest."""
    import os
    import shutil
    import subprocess

    root = tmp_path / "raglab"
    (root / "deploy").mkdir(parents=True)
    shutil.copy(ROOT / "deploy" / "deploy.sh", root / "deploy" / "deploy.sh")
    (root / "deploy" / ".env").write_text("TAG=v2.0.14\n")
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    log = tmp_path / "docker.log"
    (fake_bin / "docker").write_text(f"""#!/bin/bash
echo "$@" >> "{log}"
case "$1 $2" in
  "images --format") printf '%s\\n' ghcr.io/x/rag-pipeline-lab:v2.0.12 ghcr.io/x/rag-pipeline-lab:v2.0.13 ghcr.io/x/rag-pipeline-lab:v2.0.14 ghcr.io/x/rag-pipeline-lab-db:pg17 ;;
esac
exit 0
""")
    os.chmod(fake_bin / "docker", 0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    subprocess.run(["bash", str(root / "deploy" / "deploy.sh"), "up", "v2.0.15", "--no-pull"], check=True, env=env, capture_output=True)
    calls = log.read_text().splitlines()
    removed = [c.split()[-1] for c in calls if c.startswith("rmi")]
    assert removed == ["ghcr.io/x/rag-pipeline-lab:v2.0.12", "ghcr.io/x/rag-pipeline-lab:v2.0.13"]   # v2.0.14 (running) and v2.0.15 (incoming) kept; the db image untouched
    assert any(c.startswith("image prune") for c in calls)
    assert (root / "deploy" / ".env").read_text().strip() == "TAG=v2.0.15" and (root / "deploy" / ".previous-tag").read_text().strip() == "v2.0.14"
