"""Cross-encoder reranking over fused candidates. The reranker reads query
and chunk jointly — catching negation, plan/option distinctions, and
answers-vs-discusses — and its scores double as the abstention gauge:
best-score-below-threshold means the corpus likely lacks an answer.

Model is local (no-paid-API constraint), pre-fetched by
`huggingface_hub.snapshot_download` as an explicit setup step.
"""

import os

from raglab.retrieval import Candidate

# Rerankers under comparison; one ships. Cross-encoder scores are not
# calibrated across models, so each carries its own abstention threshold,
# derived from the golden set's answerable/unanswerable score separation
# (v1 method for bge-base: answerable best scores bottomed at 0.72, the
# absent-topic question peaked at 0.30 → midpoint 0.5. KNOWN LIMIT:
# redirect-style unanswerables score high on genuinely relevant chunks and
# are caught at the generation layer instead).
# One reranker ships. gte-reranker-modernbert-base (2025, 149M) was measured
# against it on 2026-09-08 and lost: hit@5 0.931 vs 0.966, coverage 0.787 vs
# 0.868, an entitled persona wrongly blocked (allow_answered 0.8), rerank
# p50 1964 vs 1157 ms; its answerable/unanswerable margin was 0.06 vs 0.42.
RERANKERS = {
    "bge-base": {"model": "BAAI/bge-reranker-base", "threshold": 0.5},  # 2023, 278M
}
RERANKER = os.environ.get("RAGLAB_RERANKER", "bge-base")
MODEL_NAME = RERANKERS[RERANKER]["model"]
ABSTAIN_THRESHOLD = RERANKERS[RERANKER]["threshold"]
TOP_N_OUT = 10

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, local_files_only=True)
    return _model


# What the cross-encoder reads (A/B, RAGLAB_RERANK_TEXT):
#   index — the whole search copy, template prefix and source header included
#   body  — the search copy minus the template prefix line ("This chunk is
#           from …") and, for call notes, minus the note header line (call
#           id, date, rep, reason, member, identity verification): metadata
#           the lexical and vector arms use, noise to a relevance judge.
RERANK_TEXT = os.environ.get("RAGLAB_RERANK_TEXT", "index")
_HEADER_SOURCES = ("call_note",)


def rerank_text(c: Candidate) -> str:
    text = c.index_text or c.content
    if RERANK_TEXT != "body":
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("This chunk is from"):
        lines = lines[1:]
    if c.doc_type in _HEADER_SOURCES and lines and lines[0].lstrip().startswith("CALL NOTE"):
        lines = lines[1:]
    return "\n".join(lines).strip() or text


def rerank(
    query: str, candidates: list[Candidate], top_n: int = TOP_N_OUT,
    stratify_years: tuple[int, ...] = (),
) -> list[Candidate]:
    """Returns the top_n candidates by cross-encoder score, scores attached.

    stratify_years: when the router resolved MULTIPLE years (YoY questions),
    similarity ranking alone lets one year's near-identical chunks crowd out
    the other's. Each year is scored against its own query and the years
    are interleaved by rank (latest year first), so every routed year has
    its best chunks at the top of the list.
    """
    if not candidates:
        return []
    model = _get_model()
    # Multi-year routes: each candidate is scored against ITS year's
    # year-neutral query (router.year_neutral), not the change-worded
    # question, so "changes this year" sections lose their built-in edge.
    from raglab import router as router_mod

    by_year = router_mod.year_queries(query, stratify_years)
    # Score the search copy (shorthand expanded, boilerplate suppressed,
    # identifiers canonical) — a cross-encoder reads plain language, not rep
    # shorthand. The display copy is what the payload cites.
    pairs = [(by_year.get(c.year, query), rerank_text(c)) for c in candidates]
    # sentence-transformers >= 3 applies sigmoid activation in predict();
    # scores arrive in 0..1 already.
    scores = model.predict(pairs)
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = float(score)
    ordered = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

    if len(stratify_years) < 2:
        return ordered[:top_n]

    # Years were scored against different queries (see above), so their
    # scores are not comparable: interleave by RANK within each year, latest
    # year first, so the top of every year is near the top of the list.
    lanes = {year: [c for c in ordered if c.year == year] for year in sorted(stratify_years, reverse=True)}
    picked, picked_ids = [], set()
    while len(picked) < top_n and any(lanes.values()):
        for year in list(lanes):
            if lanes[year] and len(picked) < top_n:
                c = lanes[year].pop(0)
                if c.chunk_id not in picked_ids:
                    picked.append(c)
                    picked_ids.add(c.chunk_id)
    # Chunks with no year affinity (shouldn't exist post-filter) fill the rest.
    for c in ordered:
        if len(picked) >= top_n:
            break
        if c.chunk_id not in picked_ids:
            picked.append(c)
            picked_ids.add(c.chunk_id)
    return picked[:top_n]


def abstention_verdict(reranked: list[Candidate]) -> tuple[bool, float]:
    """(should_abstain, best_score). Callers decide what to do with it;
    Phase 7 maps it to insufficient_evidence."""
    if not reranked:
        return True, 0.0
    best = reranked[0].rerank_score or 0.0
    return best < ABSTAIN_THRESHOLD, best
