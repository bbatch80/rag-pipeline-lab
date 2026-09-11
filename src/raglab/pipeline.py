"""The central query entrypoint: route -> retrieve (as persona) -> rerank ->
payload -> disclosure log. Every consumer of the funnel (explain, evals, the
MCP server) goes through here, so every retrieval is disclosed.

Persona is session identity: it maps to a real Postgres role assumed with
SET LOCAL ROLE before any search SQL runs. The engine trims rows before
ranking; this module never filters content in application code.
"""

import json
from dataclasses import dataclass
import uuid

import psycopg

from raglab.router import Route
from raglab import payload as payload_mod
from raglab import rerank, retrieval, router
from raglab.timing import Stopwatch

PERSONAS = ("public", "employee", "care_team", "member_services", "appeals")


def run_query(
    conn: psycopg.Connection,
    query: str,
    persona: str | None = None,
    source: str = "interactive",
    member_id: str | None = None,
    user_id: int | None = None,
) -> dict:
    """Returns the context payload; writes the disclosure record.
    `member_id` is the member context the surface has open (a rep's selected
    member); member-scoped sources are filtered to that member. `user_id` is
    the opaque id the edge resolved (raglab.identity): stamped into the
    disclosure row, never read — policies are role-only.

    Fail-closed (D10): payload build -> disclosure INSERT -> COMMIT -> return,
    one transaction. If the disclosure row cannot be written, no payload is
    returned and nothing is committed."""
    if persona is not None and persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected one of {PERSONAS}")

    watch = Stopwatch()
    ctx = retrieval.resolve_context(conn, member_id, query) if router.route(query).scope == "in_scope" \
        else retrieval.Context(query=query)
    probe = _probe(conn, query, persona, ctx, watch)
    with watch.stage("payload"):
        built = payload_mod.build(query, probe.decision, probe.reranked)
        built["payload_id"] = str(uuid.uuid4())
        built["persona"] = persona or "admin"
        built["member_context"] = member_id
        built["record_context"] = ctx.record
        built["unresolved_identifiers"] = ctx.unresolved
    _disclose_and_commit(conn, built, probe.reranked, source, user_id, watch)
    return built


@dataclass
class Probe:
    """One document search under one identity: the v1 funnel minus payload
    and disclosure, so a composed plan (Phase 3) can run several probes and
    disclose once."""
    decision: Route
    reranked: list
    search_query: str
    candidates: list = None  # the fused pool before reranking (for traces)


def _probe(conn, query: str, persona: str | None, ctx: retrieval.Context, watch: Stopwatch,
           decision: Route | None = None) -> Probe:
    with watch.stage("route"):
        decision = decision or router.route(query)
    reranked: list = []
    candidates: list = []
    search_query = query
    if decision.scope == "in_scope":
        # Re-identification is itself an entitlement: queries are translated
        # (name -> vault pseudonym) only for sessions entitled to the vault —
        # admin (the vault's owner) and care_team by grant — and only via the
        # owner connection, before the session drops to a persona role, which
        # cannot read the vault.
        from raglab import deid

        decision = retrieval.expand_versions(conn, decision, ctx, query)
        with watch.stage("translate"):
            if persona is None or persona == "care_team":
                search_query = deid.translate_query(conn, query)
            else:  # identifiers only: a key the caller typed is not re-identification
                search_query = deid.translate_query(conn, query, deid.IDENTIFIER_TYPES)
        # Embedding happens before the role switch: the cache table is the
        # owner's, and a vector does not depend on who is asking.
        with watch.stage("embed"):
            vector = retrieval.embed_cached(conn, search_query)
        if persona is not None:
            conn.execute(f"SET LOCAL ROLE persona_{persona}")
        try:
            with watch.stage("search"):
                candidates = retrieval.search(conn, search_query, vector, decision, member_key=ctx.member_key,
                                              record=ctx.record, embed=lambda t: retrieval.embed_cached(conn, t))
            with watch.stage("rerank"):
                reranked = rerank.rerank(search_query, candidates, stratify_years=decision.years)
        finally:
            if persona is not None:
                conn.execute("RESET ROLE")
    return Probe(decision=decision, reranked=reranked, search_query=search_query, candidates=candidates)


def _disclose_and_commit(conn, built: dict, reranked, source: str, user_id: int | None, watch: Stopwatch) -> None:
    """Fail-closed (D10): disclosure INSERT -> timings -> COMMIT, one
    transaction. If the disclosure row cannot be written, nothing is
    committed and the caller gets no payload."""
    with watch.stage("disclose"):
        try:
            _disclose(conn, built, reranked, source, user_id)
        except Exception as exc:  # fail closed: no audit row, no context
            DISCLOSURE_FAILURES["count"] += 1
            conn.rollback()
            raise RuntimeError("context withheld: disclosure record failed") from exc
    # The disclose stage is measured after the row exists; stamp the full
    # picture onto the same row before the commit that makes it real.
    built["timings"] = watch.snapshot()
    conn.execute(
        "UPDATE disclosure_log SET timings = %s WHERE payload_id = %s",
        (json.dumps(built["timings"]), built["payload_id"]),
    )
    conn.commit()


DISCLOSURE_FAILURES = {"count": 0}  # process memory: a DB that can't take the row can't take the count


def _disclose(conn, built: dict, reranked, source: str, user_id: int | None = None) -> None:
    chunks = reranked[: len(built.get("chunks", []))]
    hashes = []
    if chunks:
        rows = conn.execute(
            "SELECT d.content_hash FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.id = ANY(%s)",
            ([c.chunk_id for c in chunks],),
        ).fetchall()
        hashes = [r[0] for r in rows]
    conn.execute(
        """
        INSERT INTO disclosure_log
            (persona, source, query, payload_id, payload_status,
             chunk_ids, content_hashes, doc_titles, acl_basis, top_score,
             payload, user_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            built["persona"], source, built["query"], built["payload_id"],
            built["status"],
            [c.chunk_id for c in chunks],
            hashes,
            [c.doc_title for c in chunks],
            sorted({c.acl_tag for c in chunks}),
            built.get("confidence"),
            json.dumps(built),
            user_id,
        ),
    )
