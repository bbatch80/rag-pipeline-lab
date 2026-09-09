"""Hybrid retrieval: vector + BM25 (pg_textsearch) fused by reciprocal rank
fusion.

RRF fuses on rank only — score(d) = sum(1 / (K + rank_i(d))) — which
sidesteps score calibration entirely: cosine distances and BM25 scores are
incomparable, ranks always are. K=60 per the literature; insensitivity to K
is the feature, so it is not a tuning knob.
"""

from dataclasses import dataclass, field

import os
import re

import psycopg

from raglab.router import Route

RRF_K = 60
# Every visible source keeps at least this many candidates in the pool
# handed to the reranker, so a large source (call notes) cannot crowd a
# small one (brochures) out before reranking. Global ranking is unchanged;
# the floor only ADDS what is missing. 0 = off: on the v1 corpus no source
# is crowded out, and the extras only feed the reranker distractors (F7
# lost its rank-5 hit to an SOP scored 0.999). Switched on as a measured
# A/B when the first large source lands (Phase 1, call notes).
SOURCE_FLOOR = int(os.environ.get("RAGLAB_SOURCE_FLOOR", "0"))
# Identifier-shaped questions (member IDs, claim numbers, bulletin codes —
# 7+ chars with 3+ digits, so "30-day" doesn't count) are exact-match
# questions: the lexical list gets this weight in RRF, the vector list 1.
ID_SHAPED = re.compile(r"\b(?=(?:[A-Za-z-]*\d){3})[A-Za-z0-9-]{7,}\b")
LEXICAL_WEIGHT_ID = 2.0
# Member context. Member-scoped sources (records about one person, flagged
# in the sources registry) are searched only inside a member context and
# filtered to that member. The context is a structured field the calling
# surface supplies — the member the rep has open — or, as a convenience, a
# member ID / MRN typed in the question. Names are never resolved to a
# member: identity at a payer is ID plus date of birth, not a name. With no
# member context, member-scoped sources are not searched at all.
# How per-source BM25 lists merge into one lexical ranking:
#   rank  — each source's own rank (rank-1 everywhere gets equal RRF credit)
#   score — per-source scores sorted together (own statistics, one list)
LEXICAL_MERGE = os.environ.get("RAGLAB_LEXICAL_MERGE", "rank")
# Candidate pools: one per source (each source retrieves as if the others
# did not exist — its own vector top-100 and its own BM25 list, fused to its
# own top fused_limit; the reranker sees the union) or one global pool.
# Per source is the dominance guarantee: a large paraphrase-heavy source
# (10k call notes restating members' questions) cannot crowd brochure pages
# out before reranking. Cost: reranker input = fused_limit × visible sources.
POOLS = os.environ.get("RAGLAB_POOLS", "per_source")
_ID_TOKEN = re.compile(
    r"\b(?:MRN\s*[- ]?\s*\d{7}|M\s*[- ]?\s*\d{3}\s*[- ]?\s*\d{3}\s*[- ]?\s*\d{3}"
    r"|CLM[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4})\b", re.I)
# identifier kind -> (table, column) that maps it to a person key
_ID_LOOKUP = {
    "member_id": ("synthea.patients", "member_id"),
    "mrn": ("synthea.patients", "mrn"),
    "claim_id": ("synthea.call_log", "claim_id"),  # a claim belongs to one member
}


def resolve_member(conn: psycopg.Connection, member_id: str | None, query_text: str = "") -> str | None:
    """The person key of the member context: the structured member ID the
    surface supplied, else an identifier the caller holds typed in the
    question — a member ID, an MRN, or a claim ID (a claim belongs to one
    member). None when there is no member context. The question is the RAW
    question — translation replaces identifiers with pseudonyms."""
    from raglab import identifiers

    candidates: list[tuple[str, str]] = []
    if member_id:
        canon = identifiers.canonicalize("member_id", member_id)
        if canon is None:
            raise ValueError(f"not a valid member ID: {member_id!r}")
        candidates.append(("member_id", canon))
    for token in _ID_TOKEN.findall(query_text or ""):
        compact = re.sub(r"[\s-]", "", token)
        for kind in ("member_id", "mrn", "claim_id"):
            canon = identifiers.canonicalize(kind, compact)
            if canon:
                candidates.append((kind, canon))
    for kind, canon in candidates:
        table, column = _ID_LOOKUP[kind]
        person = "id" if table == "synthea.patients" else "patient"
        try:
            with conn.transaction():  # savepoint: a missing table must not poison the caller's transaction
                row = conn.execute(
                    f"SELECT {person} FROM {table} WHERE {column} = %s", (canon,)
                ).fetchone()
        except psycopg.errors.UndefinedTable:  # no synthea schema (CI): no member context
            return None
        if row:
            return str(row[0])
    return None


def _source_flags(conn: psycopg.Connection) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(member-scoped doc_types, event doc_types) from the sources registry."""
    rows = conn.execute(
        "SELECT doc_type, member_scoped, event FROM sources WHERE doc_type IS NOT NULL"
    ).fetchall()
    return (tuple(r[0] for r in rows if r[1]), tuple(r[0] for r in rows if r[2]))
PER_METHOD_LIMIT = 100
FUSED_LIMIT = int(os.environ.get("RAGLAB_FUSED_LIMIT", "50"))
EF_SEARCH = 40  # Phase 2 benchmark operating point


@dataclass
class Candidate:
    chunk_id: int
    content: str
    section: str
    doc_title: str
    source_path: str
    plan_code: str | None
    year: int
    acl_tag: str
    pages: list
    vector_rank: int | None
    text_rank: int | None
    rrf_score: float
    rerank_score: float | None = None
    content_hash: str = ""
    index_text: str = ""  # the search copy: what every ranking stage reads
    doc_type: str = ""
    floor: bool = False  # admitted by the per-source floor, not the global pool


def _filters(route: Route, member_key: str | None = None,
             member_scoped: tuple[str, ...] = (), events: tuple[str, ...] = ()) -> tuple[str, list]:
    clauses, params = [], []
    if route.years:
        # An edition filter: event sources (a call has a date, not an
        # edition) pass regardless of year.
        clauses.append("(c.year = ANY(%s) OR c.doc_type = ANY(%s))")
        params += [list(route.years), list(events)]
    if member_scoped:
        if member_key:
            clauses.append("(c.doc_type <> ALL(%s) OR c.member_key = %s)")
            params += [list(member_scoped), member_key]
        else:
            clauses.append("c.doc_type <> ALL(%s)")
            params.append(list(member_scoped))
    if route.plan_codes:
        # NULL plan_code = internal docs that span plans; they pass.
        clauses.append("(c.plan_code = ANY(%s) OR c.plan_code IS NULL)")
        params.append(list(route.plan_codes))
    if route.sources:
        clauses.append("c.doc_type = ANY(%s)")
        params.append(list(route.sources))
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where, params


def search(
    conn: psycopg.Connection,
    query_text: str,
    query_vector: str,
    route: Route,
    fused_limit: int = FUSED_LIMIT,
    embed=None,
    member_key: str | None = None,
) -> list[Candidate]:
    """`member_key` is the member context (retrieval.resolve_member): member-
    scoped sources are filtered to it, or skipped when it is None.

    Multi-year routes search each year separately and merge — one blended
    ranking lets the dominant year crowd the other's chunks out of the pool
    entirely, and no downstream stage can recover a chunk that never
    surfaced. Each year is searched with the year-neutral form of the
    question (router.year_neutral): the subject plus that year, no change
    language — so a prior-year benefit table is retrieved on its subject.
    `embed` turns a per-year query into a vector (default: embed_query; the
    eval's sabotage passes a junk-vector function)."""
    if len(route.years) >= 2:
        from dataclasses import replace

        from raglab import router as router_mod

        embed = embed or embed_query
        per_year = max(15, fused_limit // len(route.years))
        merged, seen = [], set()
        by_year = router_mod.year_queries(query_text, route.years)
        for year in route.years:
            sub_text = by_year.get(year, query_text)
            sub = search(
                conn, sub_text, embed(sub_text),
                replace(route, years=(year,)), fused_limit=per_year, embed=embed,
                member_key=member_key,
            )
            for candidate in sub:
                if candidate.chunk_id not in seen:
                    merged.append(candidate)
                    seen.add(candidate.chunk_id)
        return merged

    member_scoped, events = _source_flags(conn)
    where, filter_params = _filters(route, member_key, member_scoped, events)
    lexical = query_text
    lexical_weight = LEXICAL_WEIGHT_ID if ID_SHAPED.search(query_text) else 1.0
    lexical_sources = route.sources or _ingested_sources(conn)
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(EF_SEARCH),))
    # Under heavy RLS trimming a strict HNSW scan can return fewer than k
    # visible rows (the post-filter starvation problem); iterative scan keeps
    # walking the graph until enough VISIBLE results are found.
    conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
    if POOLS == "per_source":
        pool, seen_ids = [], set()
        for doc_type in lexical_sources:
            for c in _hybrid(conn, lexical, query_vector, where + " AND c.doc_type = %s",
                             filter_params + [doc_type], fused_limit, lexical_weight, (doc_type,)):
                if c.chunk_id not in seen_ids:
                    pool.append(c)
                    seen_ids.add(c.chunk_id)
    else:
        pool = _hybrid(conn, lexical, query_vector, where, filter_params, fused_limit, lexical_weight, lexical_sources)

    # Per-source floor: any visible source with fewer than SOURCE_FLOOR
    # candidates in the global pool contributes its own best few.
    present = {}
    for c in pool:
        present[c.doc_type] = present.get(c.doc_type, 0) + 1
    seen = {c.chunk_id for c in pool}
    for doc_type in (_floor_sources(conn, route) if SOURCE_FLOOR else ()):
        if present.get(doc_type, 0) >= SOURCE_FLOOR:
            continue
        extra = _hybrid(
            conn, lexical, query_vector,
            where + " AND c.doc_type = %s", filter_params + [doc_type], SOURCE_FLOOR,
            lexical_weight, (doc_type,),
        )
        for c in extra:
            if c.chunk_id not in seen:
                c.floor = True
                pool.append(c)
                seen.add(c.chunk_id)
    return pool


def _ingested_sources(conn: psycopg.Connection) -> tuple[str, ...]:
    """doc_types with a BM25 index: every ingested vector-lane source."""
    rows = conn.execute(
        "SELECT doc_type FROM sources WHERE lane IN ('vector', 'both') AND doc_type IS NOT NULL "
        "AND status = 'ingested' ORDER BY source_id"
    ).fetchall()
    return tuple(r[0] for r in rows)


def _floor_sources(conn: psycopg.Connection, route: Route) -> list[str]:
    """doc_types the floor protects: every ingested vector-lane source, or the
    routed subset. RLS still decides what each persona's floor query sees."""
    rows = conn.execute(
        "SELECT doc_type FROM sources WHERE lane IN ('vector', 'both') "
        "AND status = 'ingested' AND doc_type IS NOT NULL ORDER BY source_id"
    ).fetchall()
    types = [r[0] for r in rows]
    if route.sources:
        types = [t for t in types if t in route.sources]
    return types


def _bm25_index(doc_type: str) -> str:
    return f"chunks_bm25_{doc_type}_idx"


def _hybrid(
    conn: psycopg.Connection, lexical: str, query_vector: str,
    where: str, filter_params: list, fused_limit: int, lexical_weight: float = 1.0,
    lexical_sources: tuple[str, ...] = (),
) -> list[Candidate]:
    """One hybrid query: a global vector arm + one BM25 list PER SOURCE,
    RRF-fused (lexical list weighted by lexical_weight), top fused_limit.

    Each source has its own BM25 index (own IDF and average length), so a
    chunk's lexical rank is its rank within its source; ranks — not scores,
    which are not comparable across indexes — feed RRF. The vector arm is
    global: one embedding space, comparable everywhere."""
    # <@> is the negative BM25 score (lower = better); the index is named so
    # the filtered scan still uses it. Non-matching chunks score 0, not NULL:
    # keep only real matches (< 0) — matches sort first, so filtering after
    # LIMIT never drops a match.
    per_source = []
    txt_params: list = []
    for doc_type in lexical_sources or ("brochure",):
        per_source.append(f"""
                SELECT c.id, c.index_text <@> to_bm25query(%s, '{_bm25_index(doc_type)}') AS neg_score
                FROM chunks c
                WHERE c.doc_type = %s{where}
                ORDER BY neg_score
                LIMIT %s""")
        txt_params += [lexical, doc_type] + filter_params + [PER_METHOD_LIMIT]
    union = "\n                UNION ALL\n".join(f"({q})" for q in per_source)
    if LEXICAL_MERGE == "score":
        txt_arm = f"""
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.neg_score) AS rank
            FROM ({union}) c
            WHERE c.neg_score < 0"""
    else:
        txt_arm = "\n            UNION ALL\n".join(f"""
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.neg_score) AS rank
            FROM ({q}) c
            WHERE c.neg_score < 0""" for q in per_source)
    sql = f"""
        WITH vec AS (
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.dist) AS rank
            FROM (
                SELECT c.id, c.embedding <=> %s::vector AS dist FROM chunks c
                WHERE c.embedding IS NOT NULL{where}
                ORDER BY dist
                LIMIT %s
            ) c
        ),
        txt AS ({txt_arm}
        ),
        fused AS (
            SELECT COALESCE(v.id, t.id) AS id,
                   COALESCE(1.0 / (%s + v.rank), 0) +
                   %s * COALESCE(1.0 / (%s + t.rank), 0) AS score,
                   v.rank AS vector_rank,
                   t.rank AS text_rank
            FROM vec v FULL OUTER JOIN txt t ON v.id = t.id
        )
        SELECT c.id, c.content,
               c.metadata->>'section', d.title, d.source_path,
               c.plan_code, c.year, c.acl_tag,
               c.metadata->'pages',
               f.vector_rank, f.text_rank, f.score,
               d.content_hash, c.doc_type, c.index_text
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY f.score DESC, c.id
        LIMIT %s
    """
    params = (
        [query_vector] + filter_params + [PER_METHOD_LIMIT]
        + txt_params
        + [RRF_K, lexical_weight, RRF_K, fused_limit]
    )
    rows = conn.execute(sql, params).fetchall()
    return [
        Candidate(
            chunk_id=r[0], content=r[1], section=r[2] or "", doc_title=r[3],
            source_path=r[4], plan_code=r[5], year=r[6], acl_tag=r[7],
            pages=r[8] or [], vector_rank=r[9], text_rank=r[10],
            rrf_score=float(r[11]), content_hash=r[12], doc_type=r[13] or "",
            index_text=r[14] or "",
        )
        for r in rows
    ]


def embed_query(text: str) -> str:
    """Query vectors must come from the same model as chunk vectors."""
    from openai import OpenAI

    from raglab.embed import MODEL, _to_vector_literal

    response = OpenAI().embeddings.create(model=MODEL, input=text)
    return _to_vector_literal(response.data[0].embedding)


EMBED_CACHE = os.environ.get("RAGLAB_EMBED_CACHE", "on") != "off"


def embed_cached(conn: psycopg.Connection, text: str) -> str:
    """embed_query through the query_embeddings table (db/eval.sql). A miss
    calls the API and stores the vector; the table may not exist yet (a
    fresh database before any eval), in which case this is embed_query."""
    import hashlib

    from raglab.embed import MODEL

    if not EMBED_CACHE:
        return embed_query(text)
    key = hashlib.sha256(text.encode()).hexdigest()
    try:
        with conn.transaction():
            row = conn.execute(
                "SELECT embedding FROM query_embeddings WHERE model = %s AND text_hash = %s",
                (MODEL, key),
            ).fetchone()
    except psycopg.Error:
        return embed_query(text)
    if row:
        return row[0]
    vector = embed_query(text)
    with conn.transaction():
        conn.execute(
            "INSERT INTO query_embeddings (model, text_hash, embedding) VALUES (%s, %s, %s) "
            "ON CONFLICT DO NOTHING", (MODEL, key, vector),
        )
    return vector
