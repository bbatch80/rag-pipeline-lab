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
# Member-scoped sources (records about one person) are searched only when
# the question — or the surface's context — refers to a member: an
# identifier, or a name on the enrollment roster. A benefits question with
# no member in it does not dig through call notes. RAGLAB_MEMBER_GATE=off
# disables it (A/B).
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
MEMBER_SCOPED = ("call_note",)  # clinical notes stay wide: care managers ask population questions (P5)
MEMBER_GATE = os.environ.get("RAGLAB_MEMBER_GATE", "off") == "on"  # off: P5-style population questions name no member
_NAME_TOKEN = re.compile(r"\b[A-Z][a-z]{2,}\b")
_roster: set[str] | None = None


def member_reference(conn: psycopg.Connection, query_text: str) -> bool:
    """Does the question name a member? Identifier-shaped tokens count; so
    does any capitalized token that is a first or last name on the roster,
    or a vault pseudonym (the translated form of a name)."""
    global _roster
    if ID_SHAPED.search(query_text) or "[" in query_text:
        return True
    if _roster is None:
        try:
            rows = conn.execute("SELECT first, last FROM synthea.patients").fetchall()
            _roster = {w for f, l in rows for w in (f, l) if w}
        except psycopg.Error:
            conn.rollback()
            _roster = set()
    return any(tok in _roster for tok in _NAME_TOKEN.findall(query_text))
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
    embed=None,
) -> list[Candidate]:
    """Multi-year routes search each year separately and merge — one blended
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
            )
            for candidate in sub:
                if candidate.chunk_id not in seen:
                    merged.append(candidate)
                    seen.add(candidate.chunk_id)
        return merged

    if MEMBER_GATE and not route.sources and not member_reference(conn, query_text):
        from dataclasses import replace

        route = replace(route, sources=_non_member_sources(conn))
    where, filter_params = _filters(route)
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


def _non_member_sources(conn: psycopg.Connection) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT doc_type FROM sources WHERE lane IN ('vector', 'both') AND doc_type IS NOT NULL "
        "AND status = 'ingested' AND doc_type <> ALL(%s) ORDER BY source_id", (list(MEMBER_SCOPED),)
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
               d.content_hash, c.doc_type
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
        )
        for r in rows
    ]


def embed_query(text: str) -> str:
    """Query vectors must come from the same model as chunk vectors."""
    from openai import OpenAI

    from raglab.embed import MODEL, _to_vector_literal

    response = OpenAI().embeddings.create(model=MODEL, input=text)
    return _to_vector_literal(response.data[0].embedding)
