"""The central query entrypoint: route -> retrieve (as persona) -> rerank ->
payload -> disclosure log. Every consumer of the funnel (explain, evals, the
MCP server) goes through here, so every retrieval is disclosed.

Persona is session identity: it maps to a real Postgres role assumed with
SET LOCAL ROLE before any search SQL runs. The engine trims rows before
ranking; this module never filters content in application code.
"""

import json
import uuid

import psycopg

from raglab import payload as payload_mod
from raglab import rerank, retrieval, router
from raglab.timing import Stopwatch

PERSONAS = ("public", "employee", "care_team")


def run_query(
    conn: psycopg.Connection,
    query: str,
    persona: str | None = None,
    source: str = "interactive",
    member_id: str | None = None,
) -> dict:
    """Returns the context payload; writes the disclosure record.
    `member_id` is the member context the surface has open (a rep's selected
    member); member-scoped sources are filtered to that member."""
    if persona is not None and persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected one of {PERSONAS}")

    watch = Stopwatch()
    with watch.stage("route"):
        decision = router.route(query)
    reranked = []
    ctx = retrieval.Context(query=query)
    if decision.scope == "in_scope":
        # Re-identification is itself an entitlement: queries are translated
        # (name -> vault pseudonym) only for sessions entitled to the vault —
        # admin (the vault's owner) and care_team by grant — and only via the
        # owner connection, before the session drops to a persona role, which
        # cannot read the vault.
        from raglab import deid

        # Resolved on the raw question: translation would replace the
        # identifiers this looks for.
        ctx = retrieval.resolve_context(conn, member_id, query)
        with watch.stage("translate"):
            if persona is None or persona == "care_team":
                search_query = deid.translate_query(conn, ctx.query)
            else:  # identifiers only: a key the caller typed is not re-identification
                search_query = deid.translate_query(conn, ctx.query, deid.IDENTIFIER_TYPES)
        # Embedding happens before the role switch: the cache table is the
        # owner's, and a vector does not depend on who is asking.
        with watch.stage("embed"):
            vector = retrieval.embed_cached(conn, search_query)
        if persona is not None:
            conn.execute(f"SET LOCAL ROLE persona_{persona}")
        with watch.stage("search"):
            candidates = retrieval.search(conn, search_query, vector, decision, member_key=ctx.member_key,
                                          record=ctx.record, embed=lambda t: retrieval.embed_cached(conn, t))
        with watch.stage("rerank"):
            reranked = rerank.rerank(
                search_query, candidates, stratify_years=decision.years
            )
        if persona is not None:
            conn.execute("RESET ROLE")

    with watch.stage("payload"):
        built = payload_mod.build(query, decision, reranked)
        built["payload_id"] = str(uuid.uuid4())
        built["persona"] = persona or "admin"
        built["member_context"] = member_id
        built["record_context"] = ctx.record

    with watch.stage("disclose"):
        _disclose(conn, built, reranked, source)
    # The disclose stage is measured after the row exists; stamp the full
    # picture onto the same row before the commit that makes it real.
    conn.execute(
        "UPDATE disclosure_log SET timings = %s WHERE payload_id = %s",
        (json.dumps(watch.snapshot()), built["payload_id"]),
    )
    conn.commit()
    return built


def _disclose(conn, built: dict, reranked, source: str) -> None:
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
             payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        ),
    )
