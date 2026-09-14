"""The three context services, implemented once. The MCP server and the
web API are adapters over these functions; the CLI calls the same
pipeline entry points directly. An identity comes in as a resolved
`identity.Identity` (never a request field), becomes the planner's
`Caller`, and every leg runs as it.

One warehouse session per role is kept for the life of the process: a
composed plan closes the session it is handed after each leg, so the
planner receives a handle that ignores close.
"""

from __future__ import annotations

import threading
from typing import Callable

import psycopg

from raglab import identity, planner, snowlane
from raglab.pipeline import run_query

NO_WAREHOUSE = {
    "status": "not_authorized",
    "detail": ("This session identity has no member-data entitlement. "
               "Member claims require a claims, care-management, or actuarial role."),
}


class _SharedSession:
    """A warehouse connection handle whose close() is a no-op."""

    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return self._conn.cursor()

    def close(self):
        pass


class Warehouse:
    """One live Snowflake session per role, opened on first use."""

    def __init__(self, connect: Callable = snowlane.connect):
        self._connect = connect
        self._sessions: dict[str, object] = {}
        self._lock = threading.Lock()

    def session(self, role: str) -> _SharedSession:
        with self._lock:
            if role not in self._sessions:
                self._sessions[role] = self._connect(role=role)
            return _SharedSession(self._sessions[role])

    def close(self) -> None:
        with self._lock:
            for conn in self._sessions.values():
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 — shutting down; a dead session is fine
                    pass
            self._sessions.clear()


def caller_for(ident: identity.Identity) -> planner.Caller:
    return planner.Caller(persona=ident.persona, warehouse_role=ident.warehouse_role, user_id=ident.user_id)


def compose(conn: psycopg.Connection, ident: identity.Identity, question: str, *,
            member_id: str | None = None, case_id: str | None = None, module: str | None = None, source: str,
            warehouse: Warehouse | None = None) -> dict:
    """One question that may span sources → one composed payload (spec 1.1.0)
    under one payload id, every leg run as the identity. `member_id` and
    `case_id` are the surface's context: the member or the case it has open."""
    sf_connect = warehouse.session if warehouse is not None else None
    return planner.compose(conn, question, caller_for(ident), member_id=member_id, case_id=case_id, source=source,
                           module=module, sf_connect=sf_connect)


def search(conn: psycopg.Connection, ident: identity.Identity, query: str, *,
           member_id: str | None = None, source: str) -> dict:
    """A single document probe through the full funnel (routing, hybrid
    search, reranking, RLS trimming) → payload."""
    return run_query(conn, query, persona=ident.persona, source=source, member_id=member_id, user_id=ident.user_id)


def member_data(ident: identity.Identity, query_name: str, params: dict, *,
                warehouse: Warehouse, payload_id: str | None = None) -> dict:
    """One named catalog query as the identity's warehouse role; results name
    their masked columns. No freeform SQL crosses this boundary."""
    if ident.warehouse_role is None:
        return dict(NO_WAREHOUSE)
    return snowlane.run_named_query(warehouse.session(ident.warehouse_role), query_name,
                                    {k: v for k, v in params.items() if v is not None}, payload_id=payload_id)
