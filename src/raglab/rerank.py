"""Cross-encoder reranking over fused candidates. The reranker reads query
and chunk jointly — catching negation, plan/option distinctions, and
answers-vs-discusses — and its scores double as the abstention gauge:
best-score-below-threshold means the corpus likely lacks an answer.

Model is local (no-paid-API constraint), pre-fetched by
`huggingface_hub.snapshot_download` as an explicit setup step.
"""

from raglab.retrieval import Candidate

MODEL_NAME = "BAAI/bge-reranker-base"
TOP_N_OUT = 10
# Calibrated at the Phase 3 ablation: answerable golden questions' best
# scores bottomed at 0.72; the Ozempic-style absent-topic question peaked at
# 0.30. Midpoint 0.5 separates them. KNOWN LIMIT: redirect-style
# unanswerables (provider-directory question scored 0.81 — the corpus
# chunk pointing to the directory IS relevant, it just doesn't answer) are
# not catchable at retrieval level; generation-side groundedness (Phase 4)
# is the second line of defense.
ABSTAIN_THRESHOLD = 0.5

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, local_files_only=True)
    return _model


def rerank(
    query: str, candidates: list[Candidate], top_n: int = TOP_N_OUT
) -> list[Candidate]:
    """Returns the top_n candidates by cross-encoder score, scores attached."""
    if not candidates:
        return []
    model = _get_model()
    # sentence-transformers >= 3 applies sigmoid activation in predict();
    # scores arrive in 0..1 already.
    scores = model.predict([(query, c.content) for c in candidates])
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = float(score)
    ordered = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    return ordered[:top_n]


def abstention_verdict(reranked: list[Candidate]) -> tuple[bool, float]:
    """(should_abstain, best_score). Callers decide what to do with it;
    Phase 7 maps it to insufficient_evidence."""
    if not reranked:
        return True, 0.0
    best = reranked[0].rerank_score or 0.0
    return best < ABSTAIN_THRESHOLD, best
