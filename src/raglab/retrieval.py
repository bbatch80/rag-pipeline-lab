"""Hybrid retrieval: vector + BM25 (pg_textsearch) fused by reciprocal rank
fusion.

RRF fuses on rank only — score(d) = sum(1 / (K + rank_i(d))) — which
sidesteps score calibration entirely: cosine distances and BM25 scores are
incomparable, ranks always are. K=60 per the literature; insensitivity to K
is the feature, so it is not a tuning knob.
"""

from dataclasses import dataclass, field

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
SOURCE_FLOOR = 0
# Identifier-shaped questions (member IDs, claim numbers, bulletin codes —
# 7+ chars with 3+ digits, so "30-day" doesn't count) are exact-match
# questions: the lexical list gets this weight in RRF, the vector list 1.
ID_SHAPED = re.compile(r"\b(?=(?:[A-Za-z-]*\d){3})[A-Za-z0-9-]{7,}\b")
LEXICAL_WEIGHT_ID = 2.0
PER_METHOD_LIMIT = 100
FUSED_LIMIT = 50
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
    doc_type: str = ""
    floor: bool = False  # admitted by the per-source floor, not the global pool


def _filters(route: Route) -> tuple[str, list]:
    clauses, params = [], []
    if route.years:
        clauses.append("c.year = ANY(%s)")
        params.append(list(route.years))
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
) -> list[Candidate]:
    """Multi-year routes search each year separately and merge — one blended
    ranking lets the dominant year crowd the other's chunks out of the pool
    entirely, and no downstream stage can recover a chunk that never
    surfaced."""
    if len(route.years) >= 2:
        from dataclasses import replace

        per_year = max(15, fused_limit // len(route.years))
        merged, seen = [], set()
        for year in route.years:
            sub = search(
                conn, query_text, query_vector,
                replace(route, years=(year,)), fused_limit=per_year,
            )
            for candidate in sub:
                if candidate.chunk_id not in seen:
                    merged.append(candidate)
                    seen.add(candidate.chunk_id)
        return merged

    where, filter_params = _filters(route)
    lexical = query_text
    lexical_weight = LEXICAL_WEIGHT_ID if ID_SHAPED.search(query_text) else 1.0
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(EF_SEARCH),))
    # Under heavy RLS trimming a strict HNSW scan can return fewer than k
    # visible rows (the post-filter starvation problem); iterative scan keeps
    # walking the graph until enough VISIBLE results are found.
    conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
    pool = _hybrid(conn, lexical, query_vector, where, filter_params, fused_limit, lexical_weight)

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
            lexical_weight,
        )
        for c in extra:
            if c.chunk_id not in seen:
                c.floor = True
                pool.append(c)
                seen.add(c.chunk_id)
    return pool


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


def _hybrid(
    conn: psycopg.Connection, lexical: str, query_vector: str,
    where: str, filter_params: list, fused_limit: int, lexical_weight: float = 1.0,
) -> list[Candidate]:
    """One hybrid query: vector arm + lexical arm, RRF-fused (lexical list
    weighted by lexical_weight), top fused_limit."""
    # Lexical arm: BM25 via pg_textsearch over index_text (the search copy;
    # content is the display copy). <@> is the negative BM25 score (lower =
    # better); the index is named so the filtered scan still uses it.
    # Non-matching chunks score 0, not NULL: keep only real matches (< 0) so
    # they get no lexical rank — matches sort first, so filtering after LIMIT
    # never drops a match.
    txt_arm = f"""
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.neg_score) AS rank
            FROM (
                SELECT c.id, c.index_text <@> to_bm25query(%s, 'chunks_bm25_idx') AS neg_score
                FROM chunks c
                WHERE true{where}
                ORDER BY neg_score
                LIMIT %s
            ) c
            WHERE c.neg_score < 0"""
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
               d.content_hash, c.doc_type
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY f.score DESC, c.id
        LIMIT %s
    """
    params = (
        [query_vector] + filter_params + [PER_METHOD_LIMIT]
        + [lexical] + filter_params + [PER_METHOD_LIMIT]
        + [RRF_K, lexical_weight, RRF_K, fused_limit]
    )
    rows = conn.execute(sql, params).fetchall()
    return [
        Candidate(
            chunk_id=r[0], content=r[1], section=r[2] or "", doc_title=r[3],
            source_path=r[4], plan_code=r[5], year=r[6], acl_tag=r[7],
            pages=r[8] or [], vector_rank=r[9], text_rank=r[10],
            rrf_score=float(r[11]), content_hash=r[12], doc_type=r[13] or "",
        )
        for r in rows
    ]


def embed_query(text: str) -> str:
    """Query vectors must come from the same model as chunk vectors."""
    from openai import OpenAI

    from raglab.embed import MODEL, _to_vector_literal

    response = OpenAI().embeddings.create(model=MODEL, input=text)
    return _to_vector_literal(response.data[0].embedding)
