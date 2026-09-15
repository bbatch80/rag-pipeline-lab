"""Console reads — what the platform administrator sees: the disclosure log
in both directions (who saw what; which payloads used a document), one
payload reproduced exactly as delivered, and the health snapshot. Reads
only; nothing here changes state. The Console surface (Phase 5) renders
these; the CLI's `raglab audit` / `raglab status` report the same facts.
"""

from __future__ import annotations

import psycopg

from raglab.eval_retrieval import load_baseline
from raglab.pipeline import DISCLOSURE_FAILURES
from raglab.timing import BUDGET_P95_MS

_ROW = ("id", "asked_at", "persona", "username", "source", "query", "payload_id", "payload_status",
        "doc_titles", "acl_basis", "top_score")
_SELECT = ("SELECT d.id, d.asked_at, d.persona, u.username, d.source, d.query, d.payload_id, d.payload_status, "
           "d.doc_titles, d.acl_basis, d.top_score FROM disclosure_log d LEFT JOIN users u ON u.id = d.user_id ")


def _rows(cur) -> list[dict]:
    out = []
    for row in cur.fetchall():
        rec = dict(zip(_ROW, row))
        rec["asked_at"] = rec["asked_at"].isoformat(timespec="seconds")
        rec["payload_id"] = str(rec["payload_id"])
        rec["top_score"] = float(rec["top_score"]) if rec["top_score"] is not None else None
        out.append(rec)
    return out


def audit(conn: psycopg.Connection, *, persona: str | None = None, username: str | None = None,
          document: str | None = None, source: str | None = None, limit: int = 20) -> dict:
    """Disclosure rows, newest first, filtered by who asked (persona or
    username), which document was used (title substring), and where the
    question came from (source: web = a person at a surface; eval, trace,
    mcp… = the platform's own runs, which carry no login)."""
    clauses, params = [], []
    if source:
        clauses.append("d.source = %s")
        params.append(source)
    if persona:
        clauses.append("d.persona = %s")
        params.append(persona)
    if username:
        clauses.append("u.username = %s")
        params.append(username)
    if document:
        clauses.append("EXISTS (SELECT 1 FROM unnest(d.doc_titles) t WHERE t ILIKE %s)")
        params.append(f"%{document}%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = _rows(conn.execute(f"{_SELECT}{where} ORDER BY d.id DESC LIMIT %s", (*params, limit)))
    total, personas, users = conn.execute(
        "SELECT count(*), count(DISTINCT persona), count(DISTINCT user_id) FROM disclosure_log").fetchone()
    return {"filters": {"persona": persona, "username": username, "document": document, "source": source, "limit": limit},
            "totals": {"disclosures": total, "personas": personas, "users": users},
            "rows": rows}


def payload(conn: psycopg.Connection, payload_id: str) -> dict | None:
    """One disclosure record with the payload exactly as delivered."""
    cur = conn.execute(f"{_SELECT}WHERE d.payload_id = %s", (payload_id,))
    rows = _rows(cur)
    if not rows:
        return None
    rec = rows[0]
    rec["payload"] = conn.execute("SELECT payload FROM disclosure_log WHERE payload_id = %s", (payload_id,)).fetchone()[0]
    rec["chunk_ids"], rec["content_hashes"] = conn.execute(
        "SELECT chunk_ids, content_hashes FROM disclosure_log WHERE payload_id = %s", (payload_id,)).fetchone()
    return rec


def status(conn: psycopg.Connection) -> dict:
    """The health snapshot `raglab status` prints, as data."""
    out: dict = {"problems": []}
    out["postgres"] = conn.execute("SHOW server_version").fetchone()[0]
    ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    out["pgvector"] = ext[0] if ext else None
    if not ext:
        out["problems"].append("pgvector extension not installed")
    counts = {}
    for table in ("documents", "chunks", "quarantine", "disclosure_log"):
        if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
            out["problems"].append(f"table missing: {table}")
            continue
        counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    out["counts"] = counts
    if "chunks" in counts:
        out["chunks_without_embedding"] = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    out["synthea_patients"] = (conn.execute("SELECT count(*) FROM synthea.patients").fetchone()[0]
                               if conn.execute("SELECT to_regclass('synthea.patients')").fetchone()[0] else None)
    out["hnsw_index"] = bool(conn.execute(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'chunks_embedding_idx'").fetchone())
    if conn.execute("SELECT to_regclass('appeal_evidence')").fetchone()[0] is not None:
        drift = conn.execute(
            "SELECT (SELECT count(*) FROM documents WHERE appeal_cited IS DISTINCT FROM appeal_cited_for(title, acl_tag)), "
            "(SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.appeal_cited IS DISTINCT FROM d.appeal_cited)"
        ).fetchone()
        out["appeal_cited_drift"] = {"documents": drift[0], "chunks": drift[1]}
        if drift != (0, 0):
            out["problems"].append(f"appeal_cited flags drifted: {drift[0]} documents, {drift[1]} chunks")
    if "quarantine" in counts:
        out["quarantine"] = [{"source_path": p, "gate": g} for p, g in conn.execute(
            "SELECT source_path, gate FROM quarantine ORDER BY quarantined_at").fetchall()]
        out["problems"].extend(f"quarantined: {q['source_path']} (gate: {q['gate']})" for q in out["quarantine"])
    out["disclosure_failures"] = DISCLOSURE_FAILURES["count"]
    if DISCLOSURE_FAILURES["count"]:
        out["problems"].append(f"{DISCLOSURE_FAILURES['count']} disclosure failure(s) this process: payloads withheld")
    baseline = load_baseline()
    out["baseline"] = ({"run_id": baseline["run_id"], "git_sha": baseline.get("git_sha"), "written_at": baseline.get("written_at"),
                        "item_pass": baseline["overall"].get("item_pass"), "hit@5": baseline["overall"].get("hit@5"),
                        "guardrails_passing": len(baseline.get("guardrails_passing", []))}
                       if baseline else None)
    out["latency_budget_p95_ms"] = BUDGET_P95_MS
    out["ok"] = not out["problems"]
    return out
