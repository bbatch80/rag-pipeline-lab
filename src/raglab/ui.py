"""The surfaces (Phase 5): server-rendered pages over the Phase 4 routes.
Templates receive a payload dict and nothing else — no database, no
retrieval or entitlement logic — so `render_payload` can be exercised
from a fixture payload with nothing connected. The render contract is
fixed: status banner → evidence → plan line → footer; a draft answer is
optional, below the evidence, and never shown when evidence is missing.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from raglab import console, identity

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


def render_payload(payload: dict, *, draft_url: str | None = None, console_links: bool = False) -> str:
    """The render contract over one payload — pure: no database, no request."""
    return env.get_template("partials/payload.html").render(
        payload=payload, leg_found=leg_outcomes(payload), draft_url=draft_url, console=console_links)


def render_draft(text: str, model: str, payload_id: str) -> str:
    return env.get_template("partials/draft.html").render(text=text, model=model, payload_id=payload_id)


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

    @app.post("/ui/ask", response_class=HTMLResponse)
    def ask_fragment(question: str = Form(...), ident: identity.Identity = Depends(require_surface("ask"))):
        payload = app.state.compose_for(ident, question.strip(), "ask", None)
        return HTMLResponse(render_payload(payload, draft_url="/ui/draft", console_links="console" in ident.surfaces))

    @app.post("/ui/draft", response_class=HTMLResponse)
    def draft_fragment(payload_id: str = Form(...), ident: identity.Identity = Depends(current_identity)):
        """The optional draft: generated from the payload exactly as stored in
        the disclosure log — the user's own payload, or any for the admin —
        and never for a payload without evidence."""
        with app.state.connect() as conn:
            rec = console.payload(conn, payload_id)
        if rec is None or (rec["username"] != ident.username and "console" not in ident.surfaces):
            raise HTTPException(status_code=404, detail="no such payload for this session")
        payload = rec["payload"]
        if payload.get("status") != "ok" or not (payload.get("chunks") or payload.get("warehouse_results")):
            raise HTTPException(status_code=409, detail="no draft without evidence")
        generator = app.state.generator()
        return HTMLResponse(render_draft(generator.generate(payload), generator.name, payload_id))

    if not hasattr(app.state, "generator"):
        from raglab.generators import ClaudeGenerator

        app.state.generator = ClaudeGenerator
