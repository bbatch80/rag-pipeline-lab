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
- Every chunk carries its provenance (1.2.0): the document version label,
  the processing recipe, and the embedding model — what a hand-off team
  needs to reproduce or revoke a payload later.
"""

from datetime import datetime, timezone

from raglab.rerank import abstention_verdict
from raglab.retrieval import Candidate
from raglab.router import Route

SPEC_VERSION = "1.2.0"  # additive over 1.1.0: per-chunk provenance (document version, recipe, embedding model)


def document_version(c: Candidate) -> str:
    """The human label for WHICH version of a document a chunk came from,
    read from governed metadata (never from text). The content_hash beside
    it is the exact identity; this is what a person says out loud."""
    rec = c.record or {}
    if rec.get("version") is not None:
        label = f"v{rec['version']}"
        if rec.get("effective_from"):
            label += f" from {rec['effective_from']}"
        return label
    for key, word in (("letter_id", "letter"), ("bulletin_id", "bulletin"), ("formulary_year", "plan year")):
        if rec.get(key):
            return f"{word} {rec[key]}"
    for key in ("decided_date", "filed_date", "call_date"):
        if rec.get(key):
            return f"{key.replace('_', ' ')} {rec[key]}"
    return f"{c.year} edition" if c.year else "unversioned"


def provenance(c: Candidate) -> dict:
    """The three facts that make a logged chunk reproducible later: which
    document version, which processing recipe, which embedding model. Absent
    values are None (a corpus ingested before provenance was recorded), never
    guessed."""
    return {
        "document_version": document_version(c),
        "recipe": c.recipe or None,
        "embedding_model": c.embedding_model or None,
    }


def build(
    query: str,
    route: Route,
    reranked: list[Candidate],
    max_chunks: int = 8,
    coverage: dict | None = None,
    search: dict | None = None,
    identity_evidence: int = 0,
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
    if identity_evidence:
        # The records rule: the open member's own records answer a question
        # about them by identity; a relevance score is not the verdict.
        insufficient = False
    return {
        "spec_version": SPEC_VERSION,
        "query": query,
        **({"coverage": coverage} if coverage else {}),
        **({"search": search} if search else {}),
        **({"evidence_by_identity": identity_evidence} if identity_evidence else {}),
        "status": "insufficient_evidence" if insufficient else "ok",
        "confidence": round(best or 0.0, 4),
        "retrieved_at": retrieved_at,
        "router": {
            "years": list(route.years),
            "plan_codes": list(route.plan_codes),
            "as_of": route.as_of,
            "plan_from_enrollment": route.plan_from_enrollment,
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
                "provenance": provenance(c),
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
    router: dict | None = None,
    search: dict | None = None,
) -> dict:
    """The composed payload (spec 1.2.0, Phase 3): one question, one plan, the
    legs' results, one status. Status is worst-of-required — `ok` only if
    every required leg is `ok`; otherwise `insufficient_evidence` with
    `missing[]` naming the legs. No `partial` value: a surface derives
    partial-ness from sub_results, the status never declares it. Evidence and
    provenance only — no field expresses a determination."""
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    legs = {r["leg"]: r for r in sub_results + warehouse_results}
    missing = [name for name, leg in plan_legs(plan).items()
               if leg.get("required", True) and legs.get(name, {}).get("status") != "ok"]
    # Released (best-effort) legs never make an EMPTY payload read ok: if no leg answered, nothing did.
    answered = any(r.get("status") == "ok" for r in legs.values())
    if not missing and not answered and plan_legs(plan):
        missing = [name for name in plan_legs(plan) if legs.get(name, {}).get("status") != "ok"]
    confidences = [r.get("confidence") for r in sub_results if r.get("confidence") is not None]
    return {
        "spec_version": SPEC_VERSION,
        "query": query,
        "status": "ok" if not missing else "insufficient_evidence",
        "confidence": round(min(confidences), 4) if confidences else 0.0,
        "retrieved_at": retrieved_at,
        "router": router or next((r.get("router") for r in sub_results if r.get("router")),
                                 {"years": [], "plan_codes": [], "as_of": None, "plan_from_enrollment": False}),
        "chunks": chunks,
        "plan": plan,
        "sub_results": sub_results,
        "warehouse_results": warehouse_results,
        "missing": missing,
        "subject": subject,
        "unresolved_identifiers": unresolved,
        "as_of_defaulted": as_of_defaulted,
        **({"coverage": coverage} if coverage else {}),
        **({"search": search} if search else {}),
    }


def plan_legs(plan: dict) -> dict:
    return {leg["name"]: leg for leg in plan.get("legs", [])}
