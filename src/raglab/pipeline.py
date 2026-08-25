"""The central query entrypoint: route -> retrieve (as persona) -> rerank ->
payload -> disclosure log. Every consumer of the funnel (explain, evals, the
MCP server) goes through here, so every retrieval is disclosed.

Persona is session identity: it maps to a real Postgres role assumed with
SET LOCAL ROLE before any search SQL runs. The engine trims rows before
ranking; this module never filters content in application code.
"""

import uuid

import psycopg

from raglab import payload as payload_mod
from raglab import rerank, retrieval, router

PERSONAS = ("public", "employee", "care_team")


def run_query(
    conn: psycopg.Connection,
    query: str,
    persona: str | None = None,
    source: str = "interactive",
) -> dict:
    """Returns the context payload; writes the disclosure record."""
    if persona is not None and persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected one of {PERSONAS}")

    decision = router.route(query)
    reranked = []
    if decision.scope == "in_scope":
        # Re-identification is itself an entitlement: queries are translated
        # (name -> vault pseudonym) only for sessions entitled to the vault —
        # admin (the vault's owner) and care_team by grant — and only via the
        # owner connection, before the session drops to a persona role, which
        # cannot read the vault.
        search_query = query
        if persona is None or persona == "care_team":
            from raglab import deid

            search_query = deid.translate_query(conn, query)
        if persona is not None:
            conn.execute(f"SET LOCAL ROLE persona_{persona}")
        candidates = retrieval.search(
            conn, search_query, retrieval.embed_query(search_query), decision
        )
        reranked = rerank.rerank(
            search_query, candidates, stratify_years=decision.years
        )
        if persona is not None:
            conn.execute("RESET ROLE")

    built = payload_mod.build(query, decision, reranked)
    built["payload_id"] = str(uuid.uuid4())
    built["persona"] = persona or "admin"

    _disclose(conn, built, reranked, source)
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
             chunk_ids, content_hashes, doc_titles, acl_basis, top_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            built["persona"], source, built["query"], built["payload_id"],
            built["status"],
            [c.chunk_id for c in chunks],
            hashes,
            [c.doc_title for c in chunks],
            sorted({c.acl_tag for c in chunks}),
            built.get("confidence"),
        ),
    )
