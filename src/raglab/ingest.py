"""Document ingest with content-hash idempotency.

Contract (Phase 5 sync depends on it):
  unchanged hash -> skip
  changed hash   -> delete + reinsert (cascade removes chunks)
  new document   -> insert
  failed gate    -> quarantine, never ingest; prior good version stays
"""

import hashlib
import json
from pathlib import Path

import psycopg

from raglab import config
from raglab.chunking import chunk_elements
from raglab.gates import run_gates
from raglab.metadata import DocumentMeta, chunk_jsonb
from raglab.parsing.base import ParserBackend


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_source_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def ingest_document(
    conn: psycopg.Connection,
    pdf_path: Path,
    meta: DocumentMeta,
    backend: ParserBackend,
) -> str:
    """Returns the action taken: skipped | ingested | reingested | quarantined."""
    source_path = rel_source_path(pdf_path)
    digest = content_hash(pdf_path)

    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE source_path = %s",
        (source_path,),
    ).fetchone()
    if row is not None and row[1] == digest:
        return "skipped"

    elements = backend.parse(pdf_path)
    chunks = chunk_elements(elements)
    failures = run_gates(chunks, meta.doc_type)

    if failures:
        conn.execute("DELETE FROM quarantine WHERE source_path = %s", (source_path,))
        for failure in failures:
            conn.execute(
                "INSERT INTO quarantine (source_path, gate, detail) VALUES (%s, %s, %s)",
                (source_path, failure.gate, failure.detail),
            )
        return "quarantined"

    if row is not None:
        conn.execute("DELETE FROM documents WHERE id = %s", (row[0],))
    doc_id = conn.execute(
        """
        INSERT INTO documents (source_path, title, content_hash, acl_tag)
        VALUES (%s, %s, %s, %s) RETURNING id
        """,
        (source_path, meta.title, digest, meta.acl_tag),
    ).fetchone()[0]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, content, year, plan_code, acl_tag, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    doc_id,
                    i,
                    chunk.text,
                    meta.year,
                    meta.plan_code,
                    meta.acl_tag,
                    json.dumps(chunk_jsonb(meta, chunk)),
                )
                for i, chunk in enumerate(chunks)
            ],
        )

    conn.execute("DELETE FROM quarantine WHERE source_path = %s", (source_path,))
    return "reingested" if row is not None else "ingested"
