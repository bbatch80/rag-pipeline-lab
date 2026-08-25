"""Document ingest with content-hash idempotency.

Contract (Phase 5 sync depends on it):
  unchanged hash -> skip
  changed hash   -> delete + reinsert (cascade removes chunks)
  new document   -> insert
  failed gate    -> quarantine, never ingest; prior good version stays
"""

import hashlib
import json
import os
from pathlib import Path

import psycopg

from raglab import config
from raglab.chunking import Chunk, chunk_elements
from raglab.gates import run_gates
from raglab.metadata import DocumentMeta, chunk_jsonb
from raglab.parsing.base import ParserBackend


def _contextualize(chunks: list[Chunk], meta: DocumentMeta) -> list[Chunk]:
    """Contextual-retrieval A/B arm (RAGLAB_CONTEXTUAL=1): prepend a short
    model-written situating passage to each chunk before embedding."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI()

    def one(chunk: Chunk) -> Chunk:
        prompt = (
            f"Document: {meta.title}\nSection: {chunk.section}\n\n"
            f"Chunk:\n{chunk.text[:1500]}\n\n"
            "Write 1-2 short sentences situating this chunk within its "
            "document (which plan, year, and topic it concerns) to improve "
            "search retrieval. Reply with only the sentences."
        )
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
        )
        context = (response.choices[0].message.content or "").strip()
        return Chunk(
            text=f"{context}\n\n{chunk.text}",
            section=chunk.section,
            pages=chunk.pages,
            categories=chunk.categories,
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(one, chunks))


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

    contextual_mode = os.environ.get("RAGLAB_CONTEXTUAL")
    if contextual_mode == "1":
        chunks = _contextualize(chunks, meta)
    elif contextual_mode == "template":
        # Template arm: the situating sentence built purely from metadata we
        # already govern — no model, no API, no latency, no nondeterminism.
        chunks = [
            Chunk(
                text=(
                    f"This chunk is from {meta.title}"
                    + (f", section '{c.section}'" if c.section else "")
                    + f".\n\n{c.text}"
                ),
                section=c.section, pages=c.pages, categories=c.categories,
            )
            for c in chunks
        ]

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
