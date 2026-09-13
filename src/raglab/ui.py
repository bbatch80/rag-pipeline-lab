"""The surfaces (Phase 5): server-rendered pages over the Phase 4 routes.
Templates receive a payload dict and nothing else — no database, no
retrieval or entitlement logic — so `render_payload` can be exercised
from a fixture payload with nothing connected. The render contract is
fixed: status banner → evidence → plan line → footer. The page returns
the payload and nothing else — no generated answer (user ruling
2026-09-13; a draft answer is a possible later version).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from raglab import identifiers, identity

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

SURFACE_TITLES = {"ask": "Ask", "agent_assist": "Agent Assist", "appeals_workbench": "Appeals Workbench",
                  "analyst_view": "Analyst View", "console": "Console"}
SURFACE_BLURBS = {
    "ask": "One question, no hints. The context your role is entitled to.",
    "agent_assist": "A member on the screen: their plan, their calls, their claims — then ask.",
    "appeals_workbench": "A case on the screen: the claim, the notes it cites, the policy in effect — evidence, never a determination.",
    "analyst_view": "Named aggregate queries; masked columns labeled.",
    "console": "Sources, evaluation numbers, latency, and the disclosure log.",
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


def render_payload(payload: dict, *, console_links: bool = False) -> str:
    """The render contract over one payload — pure: no database, no request."""
    return env.get_template("partials/payload.html").render(
        payload=payload, leg_found=leg_outcomes(payload), console=console_links)


def render_workspace(*, header: dict | None, error: str | None = None, ask_url: str = "", member_id: str = "",
                     prefill: str = "", placeholder: str = "", label: str | None = None) -> str:
    """The header row after a key is committed, and the question box — pure."""
    return env.get_template("partials/workspace.html").render(
        header=header, error=error, ask_url=ask_url, member_id=member_id, prefill=prefill,
        placeholder=placeholder, label=label)


EVIDENCE_LABEL = "Evidence, never a determination: what the platform found for this case, for a person to weigh."


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
        header, error = open_record(ident, "agent_assist", "member_enrollment", "member_id", member_id)
        if error:
            return HTMLResponse(render_workspace(header=None, error=error))
        return HTMLResponse(render_workspace(
            header={**header, "title": f"Member {header['canonical']} — enrollment"},
            ask_url="/ui/agent_assist/ask", member_id=header["canonical"],
            placeholder="e.g. What did she call about last time? Is a referral needed for a specialist?"))

    @app.post("/ui/agent_assist/ask", response_class=HTMLResponse)
    def agent_assist_ask(question: str = Form(...), member_id: str = Form(...),
                         ident: identity.Identity = Depends(require_surface("agent_assist"))):
        payload = app.state.compose_for(ident, question.strip(), "agent_assist", member_id)
        return HTMLResponse(render_payload(payload, console_links="console" in ident.surfaces))

    # ---- Appeals Workbench: a case on the screen, then questions ----
    @app.get("/appeals_workbench", response_class=HTMLResponse)
    def workbench_page(request: Request, ident: identity.Identity = Depends(require_surface("appeals_workbench"))):
        return _page("appeals_workbench.html", request, _me(ident), current="appeals_workbench")

    @app.post("/ui/appeals_workbench/open", response_class=HTMLResponse)
    def workbench_open(case_id: str = Form(...), ident: identity.Identity = Depends(require_surface("appeals_workbench"))):
        header, error = open_record(ident, "appeals_workbench", "appeal_case", "case_id", case_id)
        if error:
            return HTMLResponse(render_workspace(header=None, error=error))
        # the case's member is the member context (the platform's own row, not typed);
        # the case id itself binds from the question text, so the box starts with it
        row = dict(zip(header["columns"], header["rows"][0]))
        return HTMLResponse(render_workspace(
            header={**header, "title": f"Case {header['canonical']}"},
            ask_url="/ui/appeals_workbench/ask", member_id=row.get("MEMBER_ID") or "",
            prefill=f"Case {header['canonical']}: ", placeholder="e.g. Case APL-…: why was the denial upheld?",
            label=EVIDENCE_LABEL))

    @app.post("/ui/appeals_workbench/ask", response_class=HTMLResponse)
    def workbench_ask(question: str = Form(...), member_id: str = Form(""),
                      ident: identity.Identity = Depends(require_surface("appeals_workbench"))):
        payload = app.state.compose_for(ident, question.strip(), "appeals_workbench", member_id or None)
        return HTMLResponse(render_payload(payload, console_links="console" in ident.surfaces))

    @app.post("/ui/ask", response_class=HTMLResponse)
    def ask_fragment(question: str = Form(...), ident: identity.Identity = Depends(require_surface("ask"))):
        payload = app.state.compose_for(ident, question.strip(), "ask", None)
        return HTMLResponse(render_payload(payload, console_links="console" in ident.surfaces))
