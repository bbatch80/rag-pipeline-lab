"""Example payloads for the payload page: built by the current code, so what
the page shows is always the shape the code emits."""

from __future__ import annotations

from raglab import config, payload
from raglab.retrieval import Candidate
from raglab.router import Route

SCHEMA_PATH = config.REPO_ROOT / "db" / "payload.schema.json"


def example_payload() -> dict:
    """One real build of the current spec over a fixture chunk — the example a
    consumer copies, always the shape the code emits today."""
    c = Candidate(
        chunk_id=1, content="This chunk is from GEHA FEHB 71-006 (High, Standard) 2026, section "
        "'Emergency services/accidents'.\n\nEmergency room: In-network you pay 30% of the Plan allowance "
        "after the calendar-year deductible; out-of-network 30% of the Plan allowance plus any difference "
        "between our allowance and the billed amount.",
        section="Emergency services/accidents", doc_title="GEHA FEHB 71-006 (High, Standard) 2026",
        source_path="data/raw/2026/71-006.pdf", plan_code="71-006", year=2026, acl_tag="public",
        pages=[75], vector_rank=1, text_rank=2, rrf_score=0.0323, rerank_score=0.98,
        content_hash="5af5df7a3c15a22187441014399daa5387f8a39ef64ad488451eb2dc92a1c0e4",
        doc_type="brochure", recipe="unstructured-hires|2000/1500/250|template",
        embedding_model="text-embedding-3-small",
    )
    route = Route(scope="in_scope", years=(2026,), plan_codes=("71-006",))
    return payload.build("What would a High Option member pay at the ER?", route, [c])


def example_composed_payload() -> dict:
    """The composed shape: one question, a document leg and a warehouse leg,
    one status — built by the current code so every part the payload page
    describes is on the screen."""
    question = "Why was the claim they called about on June 11, 2024 denied?"
    note = Candidate(
        chunk_id=2, content="record: call C0004417 · member [MEMBER_ID-0212] · claim CLM-8464384463 · 2024-06-11 · BENEFITS\n\n"
        "Member called about a denied claim for an office visit on 2024-06-03. Explained the provider is out of network "
        "for the member's plan; advised on the appeal process and the timely-filing window. Disposition: resolved.",
        section="", doc_title="Call C0004417 (2024-06-11)", source_path="data/internal/call_notes/C0004417.md",
        plan_code=None, year=2024, acl_tag="member_services", pages=[1], vector_rank=1, text_rank=1,
        rrf_score=0.0328, rerank_score=0.97, content_hash="9c1d3e5f7a2b4c6d8e0f1a3b5c7d9e1f2a4b6c8d0e2f4a6b8c0d2e4f6a8b0c2d",
        record={"call_id": "C0004417", "member_id": "[MEMBER_ID-0212]", "claim_id": "CLM-8464384463",
                "call_date": "2024-06-11", "reason_code": "BENEFITS", "disposition": "resolved"},
        doc_type="call_note", recipe="markdown|2000/1500/250|template|deid:tokenize:v2|abbr:v1|boiler:0.02|dedup:v2|rec:v1",
        embedding_model="text-embedding-3-small",
    )
    single = payload.build(question, Route(scope="in_scope", years=(2024,), plan_codes=("71-006",)), [note])
    chunk = single["chunks"][0]
    plan = {"shape": "compound", "origin": "model", "model": "claude-haiku-4-5-20251001", "widened": False, "enforced": [],
            "legs": [
                {"name": "call_note", "kind": "doc_probe", "text": "call about the denied claim on June 11, 2024",
                 "sources": ["call_note"], "query_name": None, "slots": [], "required": True},
                {"name": "claim_adjudication", "kind": "member_query", "text": None, "sources": [],
                 "query_name": "claim_adjudication", "slots": ["claim_id"], "required": True},
            ]}
    sub_results = [{"leg": "call_note", "status": "ok", "confidence": chunk["scores"]["rerank"],
                    "router": single["router"], "chunk_indexes": [0], "widened": False, "reason": None}]
    warehouse_results = [{"leg": "claim_adjudication", "status": "ok", "query_name": "claim_adjudication", "reason": None,
                          "columns": ["CLAIM_ID", "STATUS", "DECISION_DATE", "DENIAL_REASON", "POLICY_ID"],
                          "rows": [["CLM-8464384463", "denied", "2024-06-20", "out_of_network", None]],
                          "row_count": 1, "masked_columns": []}]
    composed = payload.compose(question, plan, sub_results, warehouse_results, [chunk], subject="M445350895",
                               unresolved=[], as_of_defaulted=False, router=single["router"])
    composed["payload_id"] = "3f2a9c1e-0000-4000-8000-000000000000"
    composed["persona"] = "member_services"
    composed["member_context"] = "M445350895"
    return composed
