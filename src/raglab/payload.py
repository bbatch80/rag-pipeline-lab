"""Context payload spec 1.0.0 — the hand-off contract between retrieval and
any consuming application (eval harness, CLI, MCP server). Semver: consumers
pin against `spec_version`; additive changes bump minor, breaking changes
bump major. The JSON Schema at db/payload.schema.json is the normative
definition; tests validate every build against it.

Key properties:
- Abstention is decided UPSTREAM and carried in `status`. The generator
  relays refusals; it does not have to be wise enough to invent them.
- Every chunk states its ACL basis (why the caller was allowed to see it)
  and its document content_hash (lineage back to the exact source version).
"""

from datetime import datetime, timezone

from raglab.rerank import abstention_verdict
from raglab.retrieval import Candidate
from raglab.router import Route

SPEC_VERSION = "1.0.0"


def build(
    query: str,
    route: Route,
    reranked: list[Candidate],
    max_chunks: int = 8,
) -> dict:
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if route.scope != "in_scope":
        return {
            "spec_version": SPEC_VERSION,
            "query": query,
            "status": "out_of_scope",
            "boundary_response": route.boundary_response,
            "retrieved_at": retrieved_at,
            "chunks": [],
        }

    insufficient, best = abstention_verdict(reranked)
    return {
        "spec_version": SPEC_VERSION,
        "query": query,
        "status": "insufficient_evidence" if insufficient else "ok",
        "confidence": round(best or 0.0, 4),
        "retrieved_at": retrieved_at,
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
                    "content_hash": c.content_hash,
                },
                "acl_basis": c.acl_tag,
                "scores": {"rrf": round(c.rrf_score, 4),
                           "rerank": round(c.rerank_score or 0.0, 4)},
            }
            for c in reranked[:max_chunks]
        ],
    }
