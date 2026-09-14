"""The surfaces (Phase 5): server-rendered pages over the Phase 4 routes.
Templates receive a payload dict and nothing else — no database, no
retrieval or entitlement logic — so `render_payload` can be exercised
from a fixture payload with nothing connected. The render contract is
fixed: status banner → evidence → plan line → footer. The page returns
the payload and nothing else — no generated answer (user ruling
2026-09-13; a draft answer is a possible later version).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from raglab import console, identifiers, identity, planner, snowlane

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

SURFACE_TITLES = {"ask": "Ask", "agent_assist": "Agent Assist", "appeals_workbench": "Appeals Workbench",
                  "analyst_view": "Analyst View", "console": "Console"}
SURFACE_BLURBS = {  # the portal tiles (user's wording, 2026-09-14)
    "ask": "Plain-language questions about plans, benefits, and policies. No context needed.",
    "agent_assist": "One member at a time: who they are, their plan, their calls and claims.",
    "appeals_workbench": "One case at a time: the claim, the record, the policy in effect, the decision's evidence.",
    "analyst_view": "Aggregate queries across members and providers; identities masked.",
    "console": "Platform health, evaluation scores, and the record of who saw what.",
}
GROUP_DESCRIPTIONS = {name: spec[3] for name, spec in identity.GROUPS.items()}

env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
env.globals.update(surface_titles=SURFACE_TITLES, surface_blurbs=SURFACE_BLURBS, group_descriptions=GROUP_DESCRIPTIONS)


def leg_outcomes(payload: dict) -> list[bool]:
    """For the plan line: did each leg of the plan return evidence? Document
    legs from sub_results (by name), warehouse legs from warehouse_results."""
    legs = (payload.get("plan") or {}).get("legs") or []
    found: dict[str, bool] = {}
    for sub in payload.get("sub_results") or []:
        found[sub.get("leg")] = sub.get("status") == "ok" and bool(sub.get("chunk_indexes"))
    for w in payload.get("warehouse_results") or []:
        found[w.get("leg")] = w.get("status") == "ok" and bool(w.get("row_count"))
    return [bool(found.get(leg.get("name"))) for leg in legs]


def answer_for(app: FastAPI, payload: dict) -> dict | None:
    """The model's answer from the payload it was given — shown FIRST on the
    surfaces (user ruling 2026-09-14), the evidence beneath it. Only when
    the platform served evidence: an insufficient or out-of-scope payload
    shows its banner and nothing generated. RAGLAB_ANSWERS=off disables."""
    if os.environ.get("RAGLAB_ANSWERS", "on") != "on":
        return None
    if payload.get("status") != "ok" or not (payload.get("chunks") or payload.get("warehouse_results")):
        return None
    generator = app.state.generator()
    started = time.perf_counter()
    try:
        text = generator.generate(payload)
    except Exception as exc:  # noqa: BLE001 — the evidence still renders; the answer says why it could not
        return {"text": None, "error": f"{type(exc).__name__}: {str(exc)[:160]}", "model": generator.name, "ms": 0}
    return {"text": text, "error": None, "model": generator.name, "ms": round((time.perf_counter() - started) * 1000)}


def _md_lite(text: str) -> str:
    """Escape the model's text, then allow exactly two things: paragraphs
    and **bold**. Nothing the model writes reaches the page as markup."""
    import html as _html
    import re as _re

    out = []
    for para in _re.split(r"\n\s*\n", text.strip()):
        safe = _html.escape(para)
        safe = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        out.append("<p>" + safe.replace("\n", "<br>") + "</p>")
    return "".join(out)


env.filters["md_lite"] = _md_lite


def render_answer(answer: dict | None) -> str:
    if answer is None:
        return ""
    return env.get_template("partials/answer.html").render(answer=answer)


def render_payload(payload: dict, *, console_links: bool = False) -> str:
    """The render contract over one payload — pure: no database, no request."""
    return env.get_template("partials/payload.html").render(
        payload=payload, leg_found=leg_outcomes(payload), console=console_links)


def render_workspace(*, opened: str = "", error: str | None = None, ask_url: str = "", member_id: str = "",
                     case_id: str = "", prefill: str = "", placeholder: str = "", label: str | None = None) -> str:
    """After a key is committed: a one-line confirmation and the question box
    carrying the key (member or case) as page state — pure. No record is
    rendered until a question is asked."""
    return env.get_template("partials/workspace.html").render(
        opened=opened, error=error, ask_url=ask_url, member_id=member_id, case_id=case_id, prefill=prefill,
        placeholder=placeholder, label=label)


EVIDENCE_LABEL = "Evidence, never a determination: what the platform found for this case, for a person to weigh."


def render_rows(result: dict, *, title: str | None = None, label: str | None = None) -> str:
    """One named-query result as a table — pure."""
    return env.get_template("partials/rows.html").render(result=result, title=title, label=label, extra_class="")


PARAM_HINTS = {"description_like": "ILIKE pattern, e.g. %asthma%", "plan_code": "e.g. 71-006", "npi": "10 digits",
               "speciality": "e.g. Cardiology (optional)", "zip_prefix": "first three digits (optional)", "limit": "rows, default 20",
               "member_id": "e.g. M767394984", "last_name": "", "first_name": "", "case_id": "e.g. APL-…", "claim_id": "e.g. CLM-…",
               "name": "organization or provider name"}


def query_params(query_name: str) -> list[tuple[str, str]]:
    """The declared parameters of a catalog query with a typing hint each."""
    spec = snowlane.NAMED_QUERIES[query_name]
    return [(name, PARAM_HINTS.get(name, "")) for name in spec.get("params", {})]


def _page(name: str, request: Request, me: dict | None, **ctx) -> HTMLResponse:
    return HTMLResponse(env.get_template(name).render(me=me, request=request, **ctx))


def mount(app: FastAPI) -> None:
    """Pages and htmx fragments over the app's identity and compose helpers."""
    from raglab.webapp import _me  # the same /me dict the JSON route returns

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    current_identity = app.state.current_identity
    require_surface = app.state.require_surface

    def me_or_none(request: Request) -> dict | None:
        try:
            return _me(current_identity(request))
        except HTTPException:
            return None

    if not hasattr(app.state, "generator"):
        from raglab.generators import ClaudeGenerator

        app.state.generator = ClaudeGenerator  # tests inject a stand-in

    @app.get("/", response_class=HTMLResponse)
    def portal(request: Request):
        me = me_or_none(request)
        if me is None:
            return RedirectResponse("/login", status_code=303)
        return _page("portal.html", request, me)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        accounts = {u: f"{display} — {GROUP_DESCRIPTIONS[group]}" for u, (display, group) in identity.SEED_USERS.items()}
        return _page("login.html", request, me_or_none(request), accounts=accounts, error=None)

    @app.post("/ui/login")
    def login_form(request: Request, username: str = Form(...), password: str = Form(...)):
        with app.state.connect() as conn:
            ident = identity.authenticate(conn, username, password)
        if ident is None:
            accounts = {u: f"{display} — {GROUP_DESCRIPTIONS[group]}" for u, (display, group) in identity.SEED_USERS.items()}
            return HTMLResponse(env.get_template("login.html").render(
                me=None, request=request, accounts=accounts, error="Unknown user or wrong password."), status_code=401)
        request.session.clear()
        request.session["username"] = ident.username
        return RedirectResponse("/", status_code=303)

    @app.post("/ui/logout")
    def logout_form(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/ask", response_class=HTMLResponse)
    def ask_page(request: Request, ident: identity.Identity = Depends(require_surface("ask"))):
        return _page("ask.html", request, _me(ident), current="ask")

    def open_record(ident: identity.Identity, surface: str, query_name: str, kind: str, value: str) -> tuple[dict | None, str | None]:
        """Validate a typed key the way the pipeline does (shape + check
        digit), then fetch the one row the platform holds for it through
        the surface's menu. Fail closed: a well-formed key that matches
        nothing is said so, and nothing else is loaded."""
        canon = identifiers.canonicalize(kind, value)
        if canon is None:
            return None, f"{value.strip()!r} is not a valid {kind.replace('_', ' ')} (shape and check digit)."
        out = app.state.member_data_for(ident, query_name, surface, {kind: canon})
        if out.get("status") != "ok":
            return None, f"This role cannot look up a {kind.replace('_', ' ')}: {out.get('status')}."
        if not out["rows"]:
            return None, f"No {kind.replace('_', ' ')} {canon} on record."
        return {**out, "canonical": canon}, None

    # ---- Agent Assist: a member on the screen, then questions ----
    @app.get("/agent_assist", response_class=HTMLResponse)
    def agent_assist_page(request: Request, ident: identity.Identity = Depends(require_surface("agent_assist"))):
        return _page("agent_assist.html", request, _me(ident), current="agent_assist")

    @app.post("/ui/agent_assist/open", response_class=HTMLResponse)
    def agent_assist_open(member_id: str = Form(...), ident: identity.Identity = Depends(require_surface("agent_assist"))):
        header, error = open_record(ident, "agent_assist", "member_profile", "member_id", member_id)
        if error:
            return HTMLResponse(render_workspace(error=error))
        return HTMLResponse(render_workspace(
            opened=f"Member {header['canonical']} is open. Ask about their plan, calls, claims, or appeals.",
            ask_url="/ui/agent_assist/ask", member_id=header["canonical"],
            placeholder="e.g. What is this member's name? What did she call about last time?"))

    @app.post("/ui/agent_assist/ask", response_class=HTMLResponse)
    def agent_assist_ask(question: str = Form(...), member_id: str = Form(...),
                         ident: identity.Identity = Depends(require_surface("agent_assist"))):
        payload = app.state.compose_for(ident, question.strip(), "agent_assist", member_id)
        return HTMLResponse(render_answer(answer_for(app, payload)) + render_payload(payload, console_links="console" in ident.surfaces))

    # ---- Appeals Workbench: a case on the screen, then questions ----
    @app.get("/appeals_workbench", response_class=HTMLResponse)
    def workbench_page(request: Request, ident: identity.Identity = Depends(require_surface("appeals_workbench"))):
        return _page("appeals_workbench.html", request, _me(ident), current="appeals_workbench")

    @app.post("/ui/appeals_workbench/open", response_class=HTMLResponse)
    def workbench_open(case_id: str = Form(...), ident: identity.Identity = Depends(require_surface("appeals_workbench"))):
        header, error = open_record(ident, "appeals_workbench", "appeal_case", "case_id", case_id)
        if error:
            return HTMLResponse(render_workspace(error=error))
        # the open case is the context: the pipeline binds its member, claim,
        # policy, and date of service to every leg (2026-09-14: the box no
        # longer needs the case id typed — "Describe this appeal" just works)
        return HTMLResponse(render_workspace(
            opened=f"Case {header['canonical']} is open. Ask about the claim, the record, or the decision.",
            ask_url="/ui/appeals_workbench/ask", case_id=header["canonical"],
            placeholder="e.g. Describe this appeal. Why was the denial upheld?",
            label=EVIDENCE_LABEL))

    @app.post("/ui/appeals_workbench/ask", response_class=HTMLResponse)
    def workbench_ask(question: str = Form(...), case_id: str = Form(...),
                      ident: identity.Identity = Depends(require_surface("appeals_workbench"))):
        payload = app.state.compose_for(ident, question.strip(), "appeals_workbench", None, case_id)
        return HTMLResponse(render_answer(answer_for(app, payload)) + render_payload(payload, console_links="console" in ident.surfaces))

    # ---- Analyst View: the module's named queries, no free text ----
    @app.get("/analyst_view", response_class=HTMLResponse)
    def analyst_page(request: Request, ident: identity.Identity = Depends(require_surface("analyst_view"))):
        from raglab.webapp import UNSCOPED, module_for

        module = module_for("analyst_view", ident.group)
        menu = tuple(snowlane.NAMED_QUERIES) if module == UNSCOPED else planner.MODULES[module]["named_queries"]
        return _page("analyst_view.html", request, _me(ident), current="analyst_view", menu=menu)

    @app.get("/ui/analyst_view/params", response_class=HTMLResponse)
    def analyst_params(query_name: str, ident: identity.Identity = Depends(require_surface("analyst_view"))):
        from raglab.webapp import UNSCOPED, module_for

        module = module_for("analyst_view", ident.group)
        menu = tuple(snowlane.NAMED_QUERIES) if module == UNSCOPED else planner.MODULES[module]["named_queries"]
        if query_name not in menu:
            raise HTTPException(status_code=400, detail=f"{query_name!r} is not on the analyst_view menu")
        return HTMLResponse(env.get_template("partials/params.html").render(
            doc=snowlane.NAMED_QUERIES[query_name]["doc"], params=query_params(query_name)))

    @app.post("/ui/analyst_view/run", response_class=HTMLResponse)
    async def analyst_run(request: Request, ident: identity.Identity = Depends(require_surface("analyst_view"))):
        form = await request.form()
        query_name = str(form.get("query_name", ""))
        params = {k: (int(v) if k == "limit" and str(v).isdigit() else str(v).strip() or None)
                  for k, v in form.items() if k != "query_name"}
        result = app.state.member_data_for(ident, query_name, "analyst_view", params)
        return HTMLResponse(render_rows(result))

    # ---- Console: health, the disclosure log, the scorecard ----
    @app.get("/console", response_class=HTMLResponse)
    def console_page(request: Request, ident: identity.Identity = Depends(require_surface("console"))):
        with app.state.connect() as conn:
            status = console.status(conn)
        return _page("console.html", request, _me(ident), current="console", status=status)

    @app.get("/ui/console/audit", response_class=HTMLResponse)
    def console_audit(username: str = "", persona: str = "", document: str = "", payload_id: str = "",
                      ident: identity.Identity = Depends(require_surface("console"))):
        if payload_id.strip():
            return RedirectResponse(f"/console/payload/{payload_id.strip()}", status_code=303)
        with app.state.connect() as conn:
            audit = console.audit(conn, username=username.strip() or None, persona=persona.strip() or None,
                                  document=document.strip() or None, limit=50)
        return HTMLResponse(env.get_template("partials/audit.html").render(audit=audit))

    @app.get("/console/payload/{payload_id}", response_class=HTMLResponse)
    def console_payload(payload_id: str, request: Request, ident: identity.Identity = Depends(require_surface("console"))):
        with app.state.connect() as conn:
            rec = console.payload(conn, payload_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no payload {payload_id!r} in the disclosure log")
        return _page("payload_page.html", request, _me(ident), current="console", rec=rec,
                     rendered=render_payload(rec["payload"]))

    @app.get("/console/dashboard")
    def console_dashboard(ident: identity.Identity = Depends(require_surface("console"))):
        """The evaluation dashboard as the last full run wrote it."""
        from raglab.dashboard import OUT_PATH

        if not OUT_PATH.exists():
            return HTMLResponse("<p>No dashboard yet: run <code>raglab eval-retrieval</code> or <code>raglab dashboard</code>.</p>")
        return FileResponse(OUT_PATH, media_type="text/html")

    @app.post("/ui/ask", response_class=HTMLResponse)
    def ask_fragment(question: str = Form(...), ident: identity.Identity = Depends(require_surface("ask"))):
        payload = app.state.compose_for(ident, question.strip(), "ask", None)
        return HTMLResponse(render_answer(answer_for(app, payload)) + render_payload(payload, console_links="console" in ident.surfaces))
