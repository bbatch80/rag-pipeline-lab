"""Two-lane MCP context server — a thin protocol adapter over the raglab
package. All retrieval, governance, and audit logic lives in the package;
this module only maps MCP tool calls onto it.

Three tools: `search_documents` (one document probe), `query_member_data`
(one named warehouse query), and `compose_context` (Phase 3: one question
in, the composed payload out — the platform plans the legs and runs them as
the session identity under ONE payload id). The two primitives stay for
agents that already know the leg.

Caller identity is SESSION CONFIG: the RAGLAB_PERSONA environment variable,
read once at server startup. It is never a tool parameter — a model cannot
claim an identity. Each identity pairs a document-lane persona (Postgres
RLS role) with a member-data-lane role (Snowflake); production maps this to
Entra ID SSO passthrough.
"""

import os

from mcp.server.mcpserver import MCPServer

from raglab import db, snowlane

# identity -> (Lane 1 Postgres persona, Lane 2 Snowflake role).
# actuary holds no internal document tier (baseline documents only) but the
# de-identified aggregate role in the warehouse. public holds no member-data
# access at all.
IDENTITIES = {
    "public": ("public", None),
    "employee": ("employee", "CLAIMS_EXAMINER"),
    "care_team": ("care_team", "CARE_MANAGER"),
    "member_services": ("member_services", "MEMBER_SERVICES_REP"),
    "appeals": ("appeals", "APPEALS_ANALYST"),
    "actuary": ("public", "ACTUARY"),
}

USER = os.environ.get("RAGLAB_USER")  # Phase 2: a seeded username, resolved through the identity tables
PERSONA = os.environ.get("RAGLAB_PERSONA", "public")
if USER is None and PERSONA not in IDENTITIES:
    raise SystemExit(
        f"RAGLAB_PERSONA={PERSONA!r} is not one of {sorted(IDENTITIES)}"
    )

mcp = MCPServer("raglab")

_sf_conn = None


def _identity(conn):
    """The session identity: RAGLAB_USER resolved through the identity tables
    (Phase 2), else the legacy RAGLAB_PERSONA mapping. Never a tool parameter."""
    from raglab import identity as identity_mod

    if USER:
        return identity_mod.resolve(conn, USER)
    lane1, role = IDENTITIES[PERSONA]
    return identity_mod.Identity(0, PERSONA, PERSONA, PERSONA, lane1, role)


def _snowflake(role: str):
    global _sf_conn
    if _sf_conn is None:
        _sf_conn = snowlane.connect(role=role)
    return _sf_conn


class _SharedSession:
    """The server keeps one warehouse session; a composed plan closes the
    session it is handed after each leg, so hand it one that ignores close."""

    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return self._conn.cursor()

    def close(self):
        pass


@mcp.tool()
def compose_context(question: str, member_id: str | None = None, module: str | None = None) -> dict:
    """Answer-ready context for ONE question that may span sources: the
    platform decides the legs (document probes over the governed corpus,
    named warehouse queries from the catalog — at most three, one pass),
    runs every leg as your session identity, and returns one composed
    payload (spec 1.1.0) under one payload id.

    Use this for a question you have not already split yourself. Use
    `search_documents` / `query_member_data` when you know the exact leg.

    `module` scopes the plan to a job-shaped menu — ask | agent_assist |
    appeals_workbench | care_management | analyst_view — the way a surface
    does; omit it only for exploration (the unscoped menu).
    `member_id` is the member context (the member the user has open).
    Identifiers in the question are resolved by the platform — never
    guessed; a well-formed ID that matches nobody is reported in
    `unresolved_identifiers`.

    HONOR THE STATUS. `ok` means every required leg returned evidence.
    `insufficient_evidence` names the legs that did not in `missing[]` —
    say what is missing; you may report what the other legs found, from
    `sub_results` / `warehouse_results`, never as a complete answer.
    `plan` shows what was looked at. Answer ONLY from chunk text and
    warehouse rows; cite source title + pages for every document fact;
    report `masked_columns` as 'not visible to your role'."""
    from raglab import planner

    with db.connect() as conn:
        ident = _identity(conn)
        caller = planner.Caller(persona=ident.persona, warehouse_role=ident.warehouse_role, user_id=ident.user_id)
        return planner.compose(conn, question, caller, member_id=member_id, source="mcp", module=module,
                               sf_connect=lambda role: _SharedSession(_snowflake(role)))


@mcp.tool()
def search_documents(query: str, member_id: str | None = None) -> dict:
    """Search the governed GEHA document corpus (brochures, internal
    operations content, clinical notes, call notes — trimmed to your session
    identity by database row-level security before ranking).

    `member_id` is the member context: the member the user has open. Records
    about one member (call notes) are searched only with a member context
    and only for that member; without it they are not searched.

    Returns a context payload (spec 1.1.0, single probe): status (ok |
    insufficient_evidence | out_of_scope), confidence, and provenance-rich
    chunks. For a question that spans sources use `compose_context`.
    HONOR THE STATUS: on insufficient_evidence say you cannot answer
    from the corpus; on out_of_scope relay boundary_response verbatim. Answer
    ONLY from chunk text and cite source title + pages for every fact."""
    from raglab.pipeline import run_query

    with db.connect() as conn:
        ident = _identity(conn)
        return run_query(conn, query, persona=ident.persona, source="mcp", member_id=member_id,
                         user_id=ident.user_id)


@mcp.tool()
def query_member_data(
    query_name: str,
    last_name: str | None = None,
    first_name: str | None = None,
    description_like: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run a NAMED governed query against member claims data, executed as
    your session's warehouse role — row access and masking policies shape
    what you see. Freeform SQL is not accepted.

    Catalog:
    - member_claims_summary(last_name, first_name?): one row per matching
      member — claim-line count, total cost, payer coverage, date span.
    - member_recent_claims(last_name, limit?): one row per claim line,
      newest first — date, encounter class, description, costs.
    - cost_by_condition(description_like): aggregate across members —
      counts and costs grouped by encounter description (ILIKE pattern,
      e.g. '%asthma%').

    Each result names its masked_columns: values there are policy-masked
    for your role — report them as 'not visible to your role'. A NULL in
    any OTHER column is genuinely absent source data, not masking."""
    with db.connect() as conn:
        role = _identity(conn).warehouse_role
    if role is None:
        return {
            "status": "not_authorized",
            "detail": (
                "This session identity has no member-data entitlement. "
                "Member claims require a claims, care-management, or "
                "actuarial role."
            ),
        }
    return snowlane.run_named_query(
        _snowflake(role), query_name,
        {"last_name": last_name, "first_name": first_name,
         "description_like": description_like, "limit": limit},
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
