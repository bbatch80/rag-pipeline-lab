"""Hybrid retrieval: vector + full-text fused by reciprocal rank fusion.

RRF fuses on rank only — score(d) = sum(1 / (K + rank_i(d))) — which
sidesteps score calibration entirely: cosine distances and BM25 scores are
incomparable, ranks always are. K=60 per the literature; insensitivity to K
is the feature, so it is not a tuning knob.
"""

from dataclasses import dataclass, field

import psycopg

from raglab.router import Route

RRF_K = 60
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


# A lexeme is "informative" if it appears in fewer than this fraction of
# chunks. Compensates for Postgres FTS lacking IDF: querying rare terms only
# stops common-word density from drowning exact-identifier matches. 1% is
# tight on purpose — moderately-common words ('id', 'deductible') belong to
# the vector arm; the lexical arm exists for near-unique identifiers.
RARE_DF_FRACTION = 0.01


def _lexical_query(conn: psycopg.Connection, query_text: str) -> str:
    """OR-of-rare-lexemes tsquery string; falls back to all lexemes when the
    query has no rare terms (or DF stats are missing)."""
    lexemes = [
        r[0]
        for r in conn.execute(
            "SELECT unnest(tsvector_to_array(to_tsvector('english', %s)))",
            (query_text,),
        ).fetchall()
    ]
    if not lexemes:
        return ""
    stats_exist = conn.execute(
        "SELECT to_regclass('lexeme_df')"
    ).fetchone()[0]
    if stats_exist:
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0] or 1
        rows = conn.execute(
            "SELECT word FROM lexeme_df WHERE word = ANY(%s) AND ndoc <= %s",
            (lexemes, max(1, int(total * RARE_DF_FRACTION))),
        ).fetchall()
        rare = [r[0] for r in rows]
        # Terms absent from the stats table never occur in the corpus at
        # all - maximally rare, keep them (typos aside, they cost nothing).
        unseen = [l for l in lexemes if l not in {r[0] for r in conn.execute(
            "SELECT word FROM lexeme_df WHERE word = ANY(%s)", (lexemes,)
        ).fetchall()}]
        chosen = rare + unseen
        if chosen:
            lexemes = chosen
    return " | ".join(f"'{l}'" for l in dict.fromkeys(lexemes))


def _filters(route: Route) -> tuple[str, list]:
    clauses, params = [], []
    if route.years:
        clauses.append("c.year = ANY(%s)")
        params.append(list(route.years))
    if route.plan_codes:
        # NULL plan_code = internal docs that span plans; they pass.
        clauses.append("(c.plan_code = ANY(%s) OR c.plan_code IS NULL)")
        params.append(list(route.plan_codes))
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
    lexical = _lexical_query(conn, query_text) or "'__nomatch__'"
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(EF_SEARCH),))
    # Under heavy RLS trimming a strict HNSW scan can return fewer than k
    # visible rows (the post-filter starvation problem); iterative scan keeps
    # walking the graph until enough VISIBLE results are found.
    conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
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
        txt AS (
            -- OR semantics over rare lexemes (see _lexical_query): AND-of-all
            -- terms returns zero rows for verbose questions, and without IDF
            -- common-word density drowns exact identifiers.
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.rank_score DESC) AS rank
            FROM (
                SELECT c.id,
                       ts_rank_cd(c.tsv, %s::tsquery) AS rank_score
                FROM chunks c
                WHERE c.tsv @@ %s::tsquery{where}
                ORDER BY rank_score DESC
                LIMIT %s
            ) c
        ),
        fused AS (
            SELECT COALESCE(v.id, t.id) AS id,
                   COALESCE(1.0 / (%s + v.rank), 0) +
                   COALESCE(1.0 / (%s + t.rank), 0) AS score,
                   v.rank AS vector_rank,
                   t.rank AS text_rank
            FROM vec v FULL OUTER JOIN txt t ON v.id = t.id
        )
        SELECT c.id, c.content,
               c.metadata->>'section', d.title, d.source_path,
               c.plan_code, c.year, c.acl_tag,
               c.metadata->'pages',
               f.vector_rank, f.text_rank, f.score
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY f.score DESC, c.id
        LIMIT %s
    """
    params = (
        [query_vector] + filter_params + [PER_METHOD_LIMIT]
        + [lexical, lexical] + filter_params + [PER_METHOD_LIMIT]
        + [RRF_K, RRF_K, fused_limit]
    )
    rows = conn.execute(sql, params).fetchall()
    return [
        Candidate(
            chunk_id=r[0], content=r[1], section=r[2] or "", doc_title=r[3],
            source_path=r[4], plan_code=r[5], year=r[6], acl_tag=r[7],
            pages=r[8] or [], vector_rank=r[9], text_rank=r[10],
            rrf_score=float(r[11]),
        )
        for r in rows
    ]


def embed_query(text: str) -> str:
    """Query vectors must come from the same model as chunk vectors."""
    from openai import OpenAI

    from raglab.embed import MODEL, _to_vector_literal

    response = OpenAI().embeddings.create(model=MODEL, input=text)
    return _to_vector_literal(response.data[0].embedding)
