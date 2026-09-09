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

from raglab import config, sources
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


def processing_recipe(backend_name: str, phi: bool = False, normalized: bool = False) -> str:
    """The recipe half of a document's identity: what would change the
    stored chunks even when the source bytes don't. PHI-bearing sources add
    the de-id mode, so changing it re-ingests exactly those documents."""
    from raglab import chunking

    contextual_mode = os.environ.get("RAGLAB_CONTEXTUAL", "template")
    recipe = (
        f"{backend_name}|{chunking.HARD_MAX}/{chunking.SOFT_MAX}/"
        f"{chunking.MERGE_UNDER}|{contextual_mode}"
    )
    if phi:
        from raglab import deid

        recipe += f"|deid:{os.environ.get('RAGLAB_DEID', 'tokenize')}:{deid.VERSION}"
    if normalized:  # dictionary version + boilerplate threshold: a change re-ingests the source
        from raglab import indexcopy

        recipe += f"|{indexcopy.RECIPE}"
    return recipe


def index_copy(text: str, source) -> str:
    """The search copy of a chunk (embedded + BM25-indexed) derived from the
    display copy. Identity for most sources; normalized sources (call notes)
    get identifier normalization, abbreviation expansion, and boilerplate
    suppression — the display copy is never touched. See raglab.indexcopy."""
    from raglab import indexcopy

    return indexcopy.normalize(text, source)


def content_hash(path: Path, recipe: str = "") -> str:
    """Fingerprint = source bytes + processing recipe. A doc is stale if
    either its file or how we process it changed."""
    digest = hashlib.sha256(path.read_bytes())
    digest.update(b"|" + recipe.encode())
    return digest.hexdigest()


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
    from raglab import indexcopy

    source = sources.load(conn).for_doc_type(meta.doc_type)
    digest = content_hash(
        pdf_path, processing_recipe(backend.name, source.phi, indexcopy.is_normalized(source))
    )

    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE source_path = %s",
        (source_path,),
    ).fetchone()
    if row is not None and row[1] == digest:
        if meta.member_key:  # backfill the person key without re-ingesting
            conn.execute(
                "UPDATE documents SET member_key = %s WHERE id = %s AND member_key IS DISTINCT FROM %s",
                (meta.member_key, row[0], meta.member_key),
            )
            conn.execute(
                "UPDATE chunks SET member_key = %s WHERE document_id = %s AND member_key IS DISTINCT FROM %s",
                (meta.member_key, row[0], meta.member_key),
            )
        return "skipped"

    elements = backend.parse(pdf_path)
    chunks = chunk_elements(elements, profile=source.chunk_profile or "section")
    failures = run_gates(chunks, source.gate_rules())

    if failures:
        conn.execute("DELETE FROM quarantine WHERE source_path = %s", (source_path,))
        for failure in failures:
            conn.execute(
                "INSERT INTO quarantine (source_path, gate, detail) VALUES (%s, %s, %s)",
                (source_path, failure.gate, failure.detail),
            )
        return "quarantined"

    # PHI de-identification runs BEFORE context/embedding: protected text
    # must never enter the embedding space or searchable corpus.
    deid_mode = os.environ.get("RAGLAB_DEID", "tokenize")
    if source.phi and deid_mode in ("mask", "tokenize"):
        from raglab import deid
        from raglab.chunking import Chunk as _Chunk

        chunks = [
            _Chunk(
                text=deid.deidentify(conn, c.text, deid_mode),
                section=c.section, pages=c.pages, categories=c.categories,
            )
            for c in chunks
        ]

    # Production default: template context (Phase 4 A/B winner — captures
    # most of LLM-contextual's coverage gain at zero cost from metadata we
    # already govern). RAGLAB_CONTEXTUAL=plain disables; =1 uses the LLM arm.
    contextual_mode = os.environ.get("RAGLAB_CONTEXTUAL", "template")
    if contextual_mode == "1":
        chunks = _contextualize(chunks, meta)
    elif contextual_mode == "template":
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
        if source.chunk_profile == "record":
            from raglab import dedup

            dedup.forget(source.key, row[0])
        conn.execute("DELETE FROM documents WHERE id = %s", (row[0],))
    doc_id = conn.execute(
        """
        INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id, member_key)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
        """,
        (source_path, meta.title, digest, meta.acl_tag, source.source_id, meta.member_key),
    ).fetchone()[0]

    # Record sources: a whole-document near-duplicate points at its original
    # and carries no chunks (never retrieved; still on disk and auditable).
    if source.chunk_profile == "record":
        from raglab import dedup

        original = dedup.check_and_add(source.key, doc_id, "\n".join(c.text for c in chunks))
        if original is not None:
            conn.execute("UPDATE documents SET duplicate_of = %s WHERE id = %s", (original, doc_id))
            conn.execute("DELETE FROM quarantine WHERE source_path = %s", (source_path,))
            return "duplicate"

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, content, index_text, year, plan_code, acl_tag,
                 doc_type, metadata, member_key)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    doc_id,
                    i,
                    chunk.text,
                    index_copy(chunk.text, source),
                    meta.year,
                    meta.plan_code,
                    meta.acl_tag,
                    meta.doc_type,
                    json.dumps(chunk_jsonb(meta, chunk)),
                    meta.member_key,
                )
                for i, chunk in enumerate(chunks)
            ],
        )

    conn.execute("DELETE FROM quarantine WHERE source_path = %s", (source_path,))
    return "reingested" if row is not None else "ingested"
