"""The deploy smoke test: log in as a seeded account, read the health
snapshot, and put one golden question through `/query`. Exit non-zero on
anything short of a served payload. Run by the release workflow after
`compose up` and by hand against any host."""

from __future__ import annotations

import os

import httpx

from raglab import ablation

DEFAULT_QUESTION_ID = "factual-01"


def run(base_url: str, username: str = "admin", password: str | None = None, question_id: str = DEFAULT_QUESTION_ID,
        timeout: float = 120.0, verify: bool | str = True) -> dict:
    """Returns the findings; raises RuntimeError with the failing step."""
    password = password or os.environ.get("RAGLAB_DEMO_PASSWORD") or "raglab-demo"
    item = next((i for i in ablation.load_golden() if i["id"] == question_id), None)
    if item is None:
        raise RuntimeError(f"no golden question {question_id!r}")
    out: dict = {"base_url": base_url, "username": username, "question_id": question_id}
    with httpx.Client(base_url=base_url, timeout=timeout, verify=verify, follow_redirects=False) as client:
        login = client.get("/login")
        if login.status_code != 200:
            raise RuntimeError(f"login page: HTTP {login.status_code}")
        me = client.post("/login", json={"username": username, "password": password})
        if me.status_code != 200:
            raise RuntimeError(f"login as {username}: HTTP {me.status_code} {me.text[:120]}")
        out["surfaces"] = me.json()["surfaces"]
        if "console" in out["surfaces"]:
            status = client.get("/status")
            if status.status_code != 200:
                raise RuntimeError(f"/status: HTTP {status.status_code}")
            health = status.json()
            out["health"] = {"ok": health["ok"], "documents": health["counts"].get("documents"),
                             "chunks": health["counts"].get("chunks"), "problems": health["problems"]}
            if not health["ok"]:
                raise RuntimeError(f"/status reports problems: {health['problems']}")
        payload = client.post("/query", json={"question": item["question"], "surface": "ask"})
        if payload.status_code != 200:
            raise RuntimeError(f"/query: HTTP {payload.status_code} {payload.text[:160]}")
        p = payload.json()
        out["payload"] = {"id": p.get("payload_id"), "status": p.get("status"), "chunks": len(p.get("chunks") or []),
                          "total_ms": (p.get("timings") or {}).get("total")}
        if p.get("status") != "ok" or not p.get("chunks"):
            raise RuntimeError(f"golden question {question_id} did not return evidence: status {p.get('status')}")
    return out
