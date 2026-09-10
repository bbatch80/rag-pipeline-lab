"""`raglab explain <golden-id>`: why a golden question hits or misses.

A miss is almost always one of three things, and this prints all three
without a script: did the expected chunk reach the candidate pool, where
did the reranker place it, and what text did the reranker actually score.
Runs the same stages as the eval (member context, translation, routing,
per-source pools, reranking) as admin."""

from dataclasses import dataclass, field

import psycopg

from raglab import ablation, deid, rerank, retrieval, router


@dataclass
class Explanation:
    qid: str
    question: str
    translated: str
    member_key: str | None
    route: dict
    pool_size: int
    pool_by_source: dict
    expected: list[dict] = field(default_factory=list)  # one per expected source
    top: list[dict] = field(default_factory=list)       # reranked top 10
    verdict: tuple[bool, float] = (False, 0.0)


def explain(conn: psycopg.Connection, qid: str, top_n: int = 10) -> Explanation:
    items = {g["id"]: g for g in ablation.load_golden()}
    if qid not in items:
        raise KeyError(f"no golden item {qid!r}")
    item = items[qid]
    question = item.get("doc_probe") or item["question"]
    ctx = retrieval.resolve_context(conn, item.get("member_id"), question)
    member_key = ctx.member_key
    translated = deid.translate_query(conn, ctx.query)
    route = retrieval.expand_versions(conn, router.route(question), ctx, question)
    vector = retrieval.embed_cached(conn, translated)
    pool = retrieval.search(conn, translated, vector, route, member_key=member_key, record=ctx.record,
                            embed=lambda t: retrieval.embed_cached(conn, t))
    ranked = rerank.rerank(translated, pool, top_n=len(pool) or 1, stratify_years=route.years)
    position = {c.chunk_id: i for i, c in enumerate(ranked)}
    by_source: dict[str, int] = {}
    for c in pool:
        by_source[c.doc_type] = by_source.get(c.doc_type, 0) + 1

    expected_specs = item.get("sources") or ([{"title": item["doc_anchor"]}] if item.get("doc_anchor") else [])
    expected = []
    for spec in expected_specs:
        if "title" in spec:
            matches = [c for c in pool if spec["title"] in c.doc_title]
        else:
            matches = [c for c in pool if ablation.is_relevant(c, [spec])]
        best = min(matches, key=lambda c: position[c.chunk_id]) if matches else None
        expected.append({
            "spec": spec,
            "in_pool": bool(matches),
            "pool_rank": min((i for i, c in enumerate(pool) if c in matches), default=None),
            "rerank_position": position[best.chunk_id] if best else None,
            "score": getattr(best, "rerank_score", None) if best else None,
            "scored_text": rerank.rerank_text(best)[:400] if best else None,
        })
    top = [{"title": c.doc_title, "score": getattr(c, "rerank_score", None), "doc_type": c.doc_type,
            "section": c.section, "text": rerank.rerank_text(c)[:160]} for c in ranked[:top_n]]
    return Explanation(
        qid=qid, question=question, translated=translated, member_key=member_key,
        route={"scope": route.scope, "years": route.years, "plan_codes": route.plan_codes, "sources": route.sources,
               "record": ctx.record},
        pool_size=len(pool), pool_by_source=by_source, expected=expected, top=top,
        verdict=rerank.abstention_verdict(ranked[:top_n]),
    )
