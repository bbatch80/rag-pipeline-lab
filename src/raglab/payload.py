"""Context payload spec v0 — the hand-off contract between retrieval and any
consuming application. Formalized (versioned, documented) at Phase 7; the
eval harness consumes it from day one so the contract is exercised, not
aspirational.

Key property: abstention is decided UPSTREAM and carried in `status`. The
generator relays refusals; it does not have to be wise enough to invent them.
"""

from raglab.rerank import ABSTAIN_THRESHOLD
from raglab.retrieval import Candidate
from raglab.router import Route

SPEC_VERSION = "0.1"


def build(
    query: str,
    route: Route,
    reranked: list[Candidate],
    max_chunks: int = 8,
) -> dict:
    if route.scope != "in_scope":
        return {
            "spec_version": SPEC_VERSION,
            "query": query,
            "status": "out_of_scope",
            "boundary_response": route.boundary_response,
            "chunks": [],
        }

    best = reranked[0].rerank_score if reranked else 0.0
    insufficient = not reranked or (best or 0.0) < ABSTAIN_THRESHOLD
    return {
        "spec_version": SPEC_VERSION,
        "query": query,
        "status": "insufficient_evidence" if insufficient else "ok",
        "confidence": round(best or 0.0, 4),
        "router": {
            "years": list(route.years),
            "plan_codes": list(route.plan_codes),
        },
        "chunks": [
            {
                "text": c.content,
                "source": {
                    "title": c.doc_title,
                    "path": c.source_path,
                    "plan_code": c.plan_code,
                    "year": c.year,
                    "pages": c.pages,
                    "section": c.section,
                },
                "scores": {"rrf": round(c.rrf_score, 4),
                           "rerank": round(c.rerank_score or 0.0, 4)},
            }
            for c in reranked[:max_chunks]
        ],
    }
