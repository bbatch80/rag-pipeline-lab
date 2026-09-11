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

SPEC_VERSION = "1.1.0"  # additive over 1.0.0: doc_type on chunk sources; composed fields (Phase 3)


def build(
    query: str,
    route: Route,
    reranked: list[Candidate],
    max_chunks: int = 8,
    coverage: dict | None = None,
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
        **({"coverage": coverage} if coverage else {}),
        "status": "insufficient_evidence" if insufficient else "ok",
        "confidence": round(best or 0.0, 4),
        "retrieved_at": retrieved_at,
        "router": {
            "years": list(route.years),
            "plan_codes": list(route.plan_codes),
            "as_of": route.as_of,
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
                    "record": c.record or {},
                    "doc_type": c.doc_type or None,
                },
                "acl_basis": c.acl_tag,
                "scores": {"rrf": round(c.rrf_score, 4),
                           "rerank": round(c.rerank_score or 0.0, 4)},
            }
            for c in reranked[:max_chunks]
        ],
    }


def compose(
    query: str,
    plan: dict,
    sub_results: list[dict],
    warehouse_results: list[dict],
    chunks: list[dict],
    subject: str | None,
    unresolved: list[dict],
    as_of_defaulted: bool,
    coverage: dict | None = None,
) -> dict:
    """The composed payload (spec 1.1.0, Phase 3): one question, one plan, the
    legs' results, one status. Status is worst-of-required — `ok` only if
    every required leg is `ok`; otherwise `insufficient_evidence` with
    `missing[]` naming the legs. No `partial` value: a surface derives
    partial-ness from sub_results, the status never declares it. Evidence and
    provenance only — no field expresses a determination."""
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    legs = {r["leg"]: r for r in sub_results + warehouse_results}
    missing = [name for name, leg in plan_legs(plan).items()
               if leg.get("required", True) and legs.get(name, {}).get("status") != "ok"]
    confidences = [r.get("confidence") for r in sub_results if r.get("confidence") is not None]
    return {
        "spec_version": SPEC_VERSION,
        "query": query,
        "status": "ok" if not missing else "insufficient_evidence",
        "confidence": round(min(confidences), 4) if confidences else 0.0,
        "retrieved_at": retrieved_at,
        "router": next((r.get("router") for r in sub_results if r.get("router")), {"years": [], "plan_codes": [], "as_of": None}),
        "chunks": chunks,
        "plan": plan,
        "sub_results": sub_results,
        "warehouse_results": warehouse_results,
        "missing": missing,
        "subject": subject,
        "unresolved_identifiers": unresolved,
        "as_of_defaulted": as_of_defaulted,
        **({"coverage": coverage} if coverage else {}),
    }


def plan_legs(plan: dict) -> dict:
    return {leg["name"]: leg for leg in plan.get("legs", [])}
