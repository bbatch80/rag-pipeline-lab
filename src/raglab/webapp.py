"""The HTTP doorway to the pipeline — the third consumer of the payload spec
after the CLI and the MCP server. A thin adapter: identity is resolved
server-side from a login session (never a request field), a route names
the surface it serves and the session's grants decide whether it opens,
and the platform maps the surface to the planner's module for that job.
No retrieval or governance logic lives here.

Prototype-grade by ruling (Phase 4 decision 2): a signed session cookie
holding the username, six seeded users, one shared demo password.
"""

from __future__ import annotations

import os
import secrets
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.sessions import SessionMiddleware

from raglab import console, context_services, db, identity, planner

SESSION_SECRET_ENV = "RAGLAB_SESSION_SECRET"
SESSION_KEY = "username"

# The five surfaces a browser can open, and the Console. A surface is a
# screen; a module is the planner's menu for a job (planner.MODULES). They
# differ in one place: the care manager opens the Agent Assist screen but
# composes under the care_management menu — same screen, data per
# entitlement. The page names the surface; the server decides the module.
SURFACES = ("ask", "agent_assist", "appeals_workbench", "analyst_view", "console")
SURFACE_MODULES: dict[tuple[str, str], str | None] = {
    ("agent_assist", "care_management"): "care_management",
    ("console", "admin"): None,  # the Console composes nothing
}

# Request fields that would name an identity or a menu — persona, role,
# warehouse_role, user_id, module. No body model declares them and every
# body model forbids extra fields, so a request carrying one is a 400.


def module_for(surface: str, group: str) -> str | None:
    """The planner module a surface composes under for a group. Default: the
    surface's own name; exceptions in SURFACE_MODULES."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; expected one of {SURFACES}")
    return SURFACE_MODULES.get((surface, group), None if surface == "console" else surface)


class Login(BaseModel):
    username: str
    password: str


class _Strict(BaseModel):
    """Request bodies reject unknown fields, so a body naming an identity
    or a menu (`persona`, `role`, `module`, ...) is a 400, never ignored."""
    model_config = ConfigDict(extra="forbid")


class Query(_Strict):
    question: str
    surface: str
    member_id: str | None = None


class Search(_Strict):
    query: str
    member_id: str | None = None


class MemberData(_Strict):
    query_name: str
    surface: str
    params: dict[str, str | int | None] = {}


def create_app(connect: Callable = db.connect, warehouse: context_services.Warehouse | None = None) -> FastAPI:
    """`connect` opens a database connection context; tests hand in the
    rolled-back fixture connection so nothing is committed. `warehouse`
    holds the process's Snowflake sessions (one per role)."""
    app = FastAPI(title="raglab", docs_url=None, redoc_url=None)
    warehouse = warehouse or context_services.Warehouse()
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.environ.get(SESSION_SECRET_ENV) or secrets.token_hex(32),  # unset: sessions end with the process
        session_cookie="raglab_session",
        same_site="lax",
        https_only=False,
    )
    app.state.connect = connect

    @app.exception_handler(RequestValidationError)
    def _bad_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """A malformed body — including one carrying an identity field — is a
        400 (decision 1), with the offending fields named."""
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    def current_identity(request: Request) -> identity.Identity:
        """The session's identity, resolved from the identity tables on every
        request (a group change takes effect at the next request)."""
        username = request.session.get(SESSION_KEY)
        if not username:
            raise HTTPException(status_code=401, detail="not logged in")
        with request.app.state.connect() as conn:
            try:
                return identity.resolve(conn, username)
            except LookupError:
                request.session.clear()
                raise HTTPException(status_code=401, detail="session user no longer exists")

    def require_surface(surface: str):
        """One dependency per route naming the surface it serves: a session
        without the grant gets 403. Convenience layer — the engines still
        enforce what data comes back."""
        if surface not in SURFACES:
            raise ValueError(f"unknown surface {surface!r}")

        def check(ident: identity.Identity = Depends(current_identity)) -> identity.Identity:
            if surface not in ident.surfaces:
                raise HTTPException(status_code=403, detail=f"surface {surface!r} is not granted to group {ident.group!r}")
            return ident

        return check

    app.state.current_identity = current_identity
    app.state.require_surface = require_surface

    @app.post("/login")
    def login(body: Login, request: Request) -> dict:
        with request.app.state.connect() as conn:
            ident = identity.authenticate(conn, body.username, body.password)
        if ident is None:
            raise HTTPException(status_code=401, detail="unknown user or wrong password")
        request.session.clear()
        request.session[SESSION_KEY] = ident.username
        return _me(ident)

    @app.post("/logout")
    def logout(request: Request) -> dict:
        request.session.clear()
        return {"logged_in": False}

    @app.get("/me")
    def me(ident: identity.Identity = Depends(current_identity)) -> dict:
        return _me(ident)

    def _surface(ident: identity.Identity, surface: str) -> str | None:
        """The grant check for a surface named in a request body, and the
        module it composes under for this identity."""
        if surface not in SURFACES:
            raise HTTPException(status_code=400, detail=f"unknown surface {surface!r}; expected one of {SURFACES}")
        require_surface(surface)(ident)
        module = module_for(surface, ident.group)
        if module is None:
            raise HTTPException(status_code=400, detail=f"surface {surface!r} composes nothing")
        return module

    def compose_for(ident: identity.Identity, question: str, surface: str, member_id: str | None) -> dict:
        """One question from a surface → the composed payload: the grant, the
        surface's module, then the same compose the MCP tool runs. The JSON
        route and the pages both come through here."""
        module = _surface(ident, surface)
        with app.state.connect() as conn:
            return context_services.compose(conn, ident, question, member_id=member_id, module=module,
                                            source="web", warehouse=warehouse)

    app.state.compose_for = compose_for

    # ---- the three context services (mirror the MCP tools one for one) ----
    @app.post("/query")
    def query(body: Query, ident: identity.Identity = Depends(current_identity)) -> dict:
        """One question from a surface → the composed payload (spec 1.1.0).
        The platform plans the legs within the surface's module and runs
        every leg as the session identity."""
        return compose_for(ident, body.question, body.surface, body.member_id)

    @app.post("/search")
    def search(body: Search, request: Request, ident: identity.Identity = Depends(current_identity)) -> dict:
        """A single document probe → payload. Any logged-in identity: the
        engine trims what it may see."""
        with request.app.state.connect() as conn:
            return context_services.search(conn, ident, body.query, member_id=body.member_id, source="web")

    @app.post("/member-data")
    def member_data(body: MemberData, ident: identity.Identity = Depends(current_identity)) -> dict:
        """One named catalog query from a surface's menu, as the session's
        warehouse role; rows plus the columns the role's policies masked."""
        module = _surface(ident, body.surface)
        menu = planner.MODULES[module]["named_queries"]
        if body.query_name not in menu:
            raise HTTPException(status_code=400, detail={"error": f"{body.query_name!r} is not on the {body.surface!r} menu",
                                                         "menu": list(menu)})
        return context_services.member_data(ident, body.query_name, body.params, warehouse=warehouse)

    # ---- Console reads (admin only) ----
    @app.get("/status")
    def status(request: Request, ident: identity.Identity = Depends(require_surface("console"))) -> dict:
        with request.app.state.connect() as conn:
            return console.status(conn)

    @app.get("/audit")
    def audit(request: Request, persona: str | None = None, username: str | None = None,
              document: str | None = None, limit: int = 20,
              ident: identity.Identity = Depends(require_surface("console"))) -> dict:
        with request.app.state.connect() as conn:
            return console.audit(conn, persona=persona, username=username, document=document, limit=min(limit, 200))

    @app.get("/payload/{payload_id}")
    def payload(payload_id: str, request: Request,
                ident: identity.Identity = Depends(require_surface("console"))) -> dict:
        with request.app.state.connect() as conn:
            rec = console.payload(conn, payload_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no payload {payload_id!r} in the disclosure log")
        return rec

    @app.get("/surfaces/{surface}")
    def surface_check(surface: str, request: Request) -> dict:
        """What a surface means for the session: opens or not, and the module
        it composes under. The frontend renders its navigation from /me; this
        answers one surface at a time and is where the grant is exercised."""
        if surface not in SURFACES:
            raise HTTPException(status_code=404, detail=f"unknown surface {surface!r}")
        ident = require_surface(surface)(current_identity(request))
        return {"surface": surface, "module": module_for(surface, ident.group)}

    from raglab import ui

    ui.mount(app)
    return app


def _me(ident: identity.Identity) -> dict:
    """What the frontend renders from: who, and which surfaces open. The
    document persona and warehouse role are shown for the demo's benefit;
    nothing accepts them back."""
    return {
        "logged_in": True,
        "username": ident.username,
        "display_name": ident.display_name,
        "group": ident.group,
        "persona": ident.persona or "admin",
        "warehouse_role": ident.warehouse_role,
        "surfaces": list(ident.surfaces),
        "modules": {s: module_for(s, ident.group) for s in ident.surfaces},
    }


app = create_app()
