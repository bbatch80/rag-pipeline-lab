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


RECORD_HEADER = "rec:v1"  # search-copy header carrying the record's keys (member records)


PARSE_STATS = {"fresh": 0, "cached": 0}  # per process: how many parses came from the cache (the receipt reports it)


def processing_recipe(backend_name: str, phi: bool = False, normalized: bool = False,
                      record: bool = False) -> str:
    """The recipe half of a document's identity: what would change the
    stored chunks even when the source bytes don't. PHI-bearing sources add
    the de-id mode, so changing it re-ingests exactly those documents."""
    from raglab import chunking

    contextual_mode = os.environ.get("RAGLAB_CONTEXTUAL", "template")
    # The table knob enters the recipe only when it splits anything: at the
    # default (whole tables, TABLE_MAX == HARD_MAX) the recipe stays byte-
    # identical to the stored corpus, so no document looks stale.
    table = f"/t{chunking.TABLE_MAX}" if chunking.TABLE_MAX < chunking.HARD_MAX else ""
    recipe = (
        f"{backend_name}|{chunking.HARD_MAX}/{chunking.SOFT_MAX}/"
        f"{chunking.MERGE_UNDER}{table}|{contextual_mode}"
    )
    if phi:
        from raglab import deid

        recipe += f"|deid:{os.environ.get('RAGLAB_DEID', 'tokenize')}:{deid.VERSION}"
    if normalized:  # dictionary version + boilerplate threshold: a change re-ingests the source
        from raglab import indexcopy

        recipe += f"|{indexcopy.RECIPE}"
    if record:  # the record's keys ride in every chunk's search copy
        recipe += f"|{RECORD_HEADER}"
    return recipe


_RECORD_KEYS = (("case_id", "CASE_ID", "case"), ("member_id", "MEMBER_ID", "member"),
                ("claim_id", "CLAIM_ID", "claim"), ("call_id", None, "call"))


def record_header(conn, record: dict, deid_mode: str) -> str:
    """One line for the search copy of every chunk of a member record: the
    record's keys, as the vault's pseudonyms (the same tokens a translated
    question carries). A section chunk then ranks on its content AND on
    whose record it is — without it, only the header section holds the
    identifiers and every other section scores near zero against an
    identifier-shaped question. Metadata (routing) and this line (ranking)
    come from the record row, never from the text. The display copy is
    untouched."""
    parts = []
    for key, entity, label in _RECORD_KEYS:
        value = record.get(key)
        if not value:
            continue
        if entity and deid_mode in ("mask", "tokenize"):
            from raglab import deid

            value = f"[{entity}]" if deid_mode == "mask" else deid._pseudonym(conn, entity, value)
        parts.append(f"{label} {value}")
    return ("record: " + "  ".join(parts)) if parts else ""


def rebuild_search_copy(conn, registry, source_keys: tuple[str, ...] = ()) -> dict:
    """Recompute every chunk's search copy (index_text) from what is already
    stored — the de-identified content and the record metadata — and clear
    the embedding of every chunk whose copy changed, so `raglab embed`
    re-embeds only those. This is the tool for a change to a DERIVED copy
    (dictionary version, boilerplate threshold, record header): the parse
    and de-id steps produced the same content last time and are not run
    again. Source bytes, parser or de-id changes still go through the
    processing recipe (a full re-ingest)."""
    from raglab.internal_corpus import manifest_meta

    deid_mode = os.environ.get("RAGLAB_DEID", "tokenize")
    stats = {}
    for source in registry.all:
        if source.lane not in ("vector", "both") or source.status != "ingested":
            continue
        if source_keys and source.key not in source_keys:
            continue
        metas = manifest_meta(source)
        changed = 0
        rows = conn.execute(
            "SELECT c.id, c.content, c.index_text, d.title FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.source_id = %s", (source.source_id,)
        ).fetchall()
        for chunk_id, content, current, title in rows:
            m = metas.get(title, {})
            meta = DocumentMeta(carrier="", plan_code=None, plan_options=(), program="", year=0,
                                doc_type=source.doc_type or "", acl_tag=source.acl_tag, effective_date="",
                                title=title, record=m.get("record") or {})
            header = search_copy_header(conn, source, meta, deid_mode)
            new = (header + "\n" if header else "") + index_copy(content, source)
            if new != current:
                conn.execute("UPDATE chunks SET index_text = %s, embedding = NULL WHERE id = %s", (new, chunk_id))
                changed += 1
        stats[source.key] = {"chunks": len(rows), "rebuilt": changed}
    return stats


def search_copy_header(conn, source, meta: DocumentMeta, deid_mode: str) -> str:
    """The record header for a chunk's search copy — only for records chunked
    by section. A one-chunk record (call notes) already carries its keys in
    its own text; adding them again only dilutes it (measured: a borderline
    call note fell from 0.195 to 0.067)."""
    if not meta.record or source.chunk_profile == "record":
        return ""
    return record_header(conn, meta.record, deid_mode)


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
        pdf_path, processing_recipe(backend.name, source.phi, indexcopy.is_normalized(source), bool(meta.record))
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
        if meta.record:  # the record's fields ride as chunk metadata; refresh without re-ingesting
            conn.execute(
                "UPDATE chunks SET metadata = metadata || jsonb_build_object('record', %s::jsonb) "
                "WHERE document_id = %s AND metadata->'record' IS DISTINCT FROM %s::jsonb",
                (json.dumps(meta.record), row[0], json.dumps(meta.record)),
            )
        return "skipped"

    from raglab import parsecache

    elements, from_cache = parsecache.parse(backend, pdf_path)
    PARSE_STATS["cached" if from_cache else "fresh"] += 1
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

    header = search_copy_header(conn, source, meta, deid_mode)
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
                    (header + "\n" if header else "") + index_copy(chunk.text, source),
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
