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
    "bge-base": {
        "model": "BAAI/bge-reranker-base",  # 2023, 278M
        "threshold": 0.5,  # prose sources (calibrated on brochures, Phase 0)
        # Records: a terse, de-identified note scores lower in absolute
        # terms even when it is the answer. Calibrated on the call-note
        # golden slice (2026-09-09, 10 items): correct notes 0.16–0.99,
        # unrelated notes ≈ 0.00–0.01; a threshold of 0.1 sits under every
        # answered item with margin and above the noise floor.
        "thresholds": {"call_note": 0.1, "appeal": 0.1},  # records: same bar (appeal chunks carry the record header)
    },
}
RERANKER = os.environ.get("RAGLAB_RERANKER", "bge-base")
MODEL_NAME = RERANKERS[RERANKER]["model"]
ABSTAIN_THRESHOLD = RERANKERS[RERANKER]["threshold"]
ABSTAIN_BY_SOURCE = RERANKERS[RERANKER].get("thresholds", {})


def threshold_for(doc_type: str) -> float:
    return ABSTAIN_BY_SOURCE.get(doc_type, ABSTAIN_THRESHOLD)
TOP_N_OUT = 10

_model = None


# ---------------------------------------------------------------- cache
# A cross-encoder score is a pure function of (model, query text, chunk
# text). The cache memoizes exactly that: the model name + the weight
# snapshot it loaded, the final ranking query string, the sha256 of the
# exact text scored (prefix + record header + search copy, per text mode).
# No chunk id, no timestamp — a re-ingest, a search-copy rebuild, a header
# rule, a query-side change, or a model swap all change the key and miss
# honestly. Thresholds are applied after scoring and are not in the key.
RERANK_CACHE = os.environ.get("RAGLAB_RERANK_CACHE", "on") != "off"
CACHE_STATS = {"hits": 0, "misses": 0}
_cache_conn = None
_model_key: str | None = None


def model_key() -> str:
    """MODEL_NAME plus the weight snapshot revision the local cache holds."""
    global _model_key
    if _model_key is None:
        rev = "local"
        try:
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_id == MODEL_NAME:
                    revs = sorted(repo.revisions, key=lambda r: r.last_modified)
                    rev = revs[-1].commit_hash[:12] if revs else "local"
        except Exception:  # no hub cache metadata: still keyed on the name
            pass
        _model_key = f"{MODEL_NAME}@{rev}"
    return _model_key


def _cache_connection():
    """An owner connection of its own (autocommit): the pipeline's session
    may be running under a persona role that cannot write. Tests patch
    this to hand in the rolled-back fixture connection."""
    global _cache_conn
    if _cache_conn is None or _cache_conn.closed:
        from raglab import db

        _cache_conn = db.connect()
        _cache_conn.autocommit = True
    return _cache_conn


def score_pairs(model, pairs: list[tuple[str, str]], conn=None) -> list[float]:
    """model.predict through the rerank_scores table: hits are read, misses
    are predicted in ONE batched call and stored. Without the table (a
    fresh database before any eval) or with the cache off, plain predict."""
    import hashlib

    if not RERANK_CACHE or not pairs:
        return [float(x) for x in model.predict(pairs)] if pairs else []
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    keys = [(h(q), h(t)) for q, t in pairs]
    mk = model_key()
    try:
        conn = conn or _cache_connection()
        with conn.transaction():
            rows = conn.execute(
                "SELECT query_hash, text_hash, score FROM rerank_scores "
                "WHERE model = %s AND query_hash = ANY(%s) AND text_hash = ANY(%s)",
                (mk, sorted({q for q, _ in keys}), sorted({t for _, t in keys})),
            ).fetchall()
    except Exception:  # no table / no database: score everything
        return [float(x) for x in model.predict(pairs)]
    known = {(q, t): float(s) for q, t, s in rows}
    scores: list[float | None] = [known.get(k) for k in keys]
    miss_idx = [i for i, s in enumerate(scores) if s is None]
    CACHE_STATS["hits"] += len(pairs) - len(miss_idx)
    CACHE_STATS["misses"] += len(miss_idx)
    if miss_idx:
        fresh = [float(x) for x in model.predict([pairs[i] for i in miss_idx])]
        for i, s in zip(miss_idx, fresh, strict=True):
            scores[i] = s
        try:
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO rerank_scores (model, query_hash, text_hash, score) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    [(mk, keys[i][0], keys[i][1], s) for i, s in zip(miss_idx, fresh, strict=True)],
                )
        except Exception:
            pass  # a cache write failure never fails a query
    return [float(s) for s in scores]  # type: ignore[arg-type]


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, local_files_only=True)
    return _model


# What the cross-encoder reads (A/B, RAGLAB_RERANK_TEXT):
#   index — the whole search copy, template prefix and source header included
#   header — for call notes, minus the note header line (call id, date,
#           rep, reason, member, identity verification): metadata the
#           lexical and vector arms use, noise to a relevance judge
#   body  — header, and minus the template prefix line ("This chunk is
#           from …") for every source
#   max   — index AND the header-less body for call notes, higher score wins
RERANK_TEXT = os.environ.get("RAGLAB_RERANK_TEXT", "max")
_HEADER_SOURCES = ("call_note", "appeal")
_HEADER_PREFIXES = ("record:", "CALL NOTE", "Appeal case", "Case ", "GEHA APPEALS DETERMINATION", "Case:", "Member:")


def record_body(c: Candidate) -> str:
    """A record's search copy without its header line."""
    text = c.index_text or c.content
    lines = [l for l in text.split("\n") if not l.lstrip().startswith(_HEADER_PREFIXES)]
    return "\n".join(lines).strip() or text


def rerank_text(c: Candidate) -> str:
    text = c.index_text or c.content
    if RERANK_TEXT not in ("body", "header"):
        return text
    lines = text.split("\n")
    if RERANK_TEXT == "body" and lines and lines[0].startswith("This chunk is from"):
        lines = lines[1:]
    if c.doc_type in _HEADER_SOURCES:
        lines = [l for l in lines if not l.lstrip().startswith(_HEADER_PREFIXES)]
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
    scores = score_pairs(model, pairs)
    if RERANK_TEXT == "max":
        # Records (call notes) are scored twice — the whole search copy and
        # the body without the note header — and take the higher. The
        # header's member/reason tokens carry an identifier-shaped question;
        # for a question with no identifier the same header buries a
        # near-verbatim body (0.99 body-only vs 0.01 with it). Cost: a
        # second pair per call note in the pool (a handful under a member
        # context).
        idx = [i for i, c in enumerate(candidates) if c.doc_type in _HEADER_SOURCES]
        if idx:
            bodies = [(pairs[i][0], record_body(candidates[i])) for i in idx]
            for i, score in zip(idx, score_pairs(model, bodies), strict=True):
                scores[i] = max(scores[i], float(score))
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = score
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
    """(should_abstain, best_score). Abstain when no candidate clears its
    own source's threshold (threshold_for). Callers decide what to do with
    it; the payload maps it to insufficient_evidence."""
    if not reranked:
        return True, 0.0
    best = reranked[0].rerank_score or 0.0
    cleared = any((c.rerank_score or 0.0) >= threshold_for(c.doc_type) for c in reranked)
    return not cleared, best
