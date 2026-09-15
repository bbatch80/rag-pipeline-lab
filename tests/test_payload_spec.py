"""Payload spec 1.0.0 contract: every shape the builder can emit validates
against the normative JSON Schema (db/payload.schema.json). Offline — no
database, no APIs."""

import json

import jsonschema
import pytest

from raglab import config, payload
from raglab.retrieval import Candidate
from raglab.router import Route

SCHEMA = json.loads((config.REPO_ROOT / "db" / "payload.schema.json").read_text())


def _candidate(score: float, tag: str = "public") -> Candidate:
    return Candidate(
        chunk_id=1, content="chunk text", section="Deductible",
        doc_title="GEHA FEHB 71-006 (High, Standard) 2026",
        source_path="data/raw/2026/71-006.pdf", plan_code="71-006",
        year=2026, acl_tag=tag, pages=[14], vector_rank=1, text_rank=None,
        rrf_score=0.03, rerank_score=score, content_hash="abc123",
    )


IN_SCOPE = Route(scope="in_scope", years=(2026,), plan_codes=("71-006",))


def test_ok_payload_validates():
    built = payload.build("q", IN_SCOPE, [_candidate(0.9)])
    jsonschema.validate(built, SCHEMA)
    assert built["status"] == "ok"
    chunk = built["chunks"][0]
    assert chunk["acl_basis"] == "public"
    assert chunk["source"]["content_hash"] == "abc123"


def test_abstention_fixture_validates():
    built = payload.build("q", IN_SCOPE, [_candidate(0.2)])
    jsonschema.validate(built, SCHEMA)
    assert built["status"] == "insufficient_evidence"


def test_empty_results_abstain():
    built = payload.build("q", IN_SCOPE, [])
    jsonschema.validate(built, SCHEMA)
    assert built["status"] == "insufficient_evidence"


def test_out_of_scope_carries_boundary():
    route = Route(scope="out_of_domain", boundary_response="GEHA only.")
    built = payload.build("q", route, [])
    jsonschema.validate(built, SCHEMA)
    assert built["status"] == "out_of_scope"
    assert built["boundary_response"] == "GEHA only."


def test_schema_rejects_missing_acl_basis():
    built = payload.build("q", IN_SCOPE, [_candidate(0.9)])
    del built["chunks"][0]["acl_basis"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(built, SCHEMA)


def test_composed_payload_validates_and_is_worst_of_required():
    plan = {"shape": "compound", "origin": "caller", "model": None, "widened": False, "legs": [
        {"name": "docs", "kind": "doc_probe", "text": "q", "sources": ["appeal"], "query_name": None, "slots": [], "required": True},
        {"name": "adj", "kind": "member_query", "text": None, "sources": [], "query_name": "claim_adjudication", "slots": ["claim_id"], "required": True},
    ]}
    docs_ok = {"leg": "docs", "status": "ok", "confidence": 0.9, "router": {"years": [2026], "plan_codes": [], "as_of": None}, "chunk_indexes": [0], "widened": False, "reason": None}
    adj_missing = {"leg": "adj", "status": "not_executed", "query_name": "claim_adjudication", "reason": "no warehouse identity"}
    chunk = payload.build("q", IN_SCOPE, [_candidate(0.9)])["chunks"][0]
    composed = payload.compose("q", plan, [docs_ok], [adj_missing], [chunk], subject="M344317862", unresolved=[], as_of_defaulted=False)
    jsonschema.validate(composed, SCHEMA)
    assert composed["spec_version"] == "1.2.0" and composed["status"] == "insufficient_evidence" and composed["missing"] == ["adj"]
    adj_ok = {**adj_missing, "status": "ok", "reason": None, "columns": ["CLAIM_ID"], "rows": [["CLM-1"]], "row_count": 1, "masked_columns": []}
    composed = payload.compose("q", plan, [docs_ok], [adj_ok], [chunk], subject=None, unresolved=[{"kind": "member_id", "value": "M999900004"}], as_of_defaulted=True)
    jsonschema.validate(composed, SCHEMA)
    assert composed["status"] == "ok" and composed["missing"] == [] and composed["as_of_defaulted"] is True
    assert "partial" not in {composed["status"]}  # no partial status: derived by the surface, never declared


def test_provenance_on_every_chunk():
    c = _candidate(0.9)
    c.recipe = "unstructured-hi_res|1600/900/200|template"
    c.embedding_model = "text-embedding-3-small"
    built = payload.build("q", IN_SCOPE, [c])
    jsonschema.validate(built, SCHEMA)
    prov = built["chunks"][0]["provenance"]
    assert prov == {"document_version": "2026 edition", "recipe": c.recipe, "embedding_model": "text-embedding-3-small"}


def test_provenance_never_guesses_when_unrecorded():
    built = payload.build("q", IN_SCOPE, [_candidate(0.9)])  # no recipe, no model on the candidate
    jsonschema.validate(built, SCHEMA)
    prov = built["chunks"][0]["provenance"]
    assert prov["recipe"] is None and prov["embedding_model"] is None


@pytest.mark.parametrize("record,year,label", [
    ({"version": 2, "effective_from": "2026-01-01", "status": "current"}, 2026, "v2 from 2026-01-01"),
    ({"letter_id": "2023-01", "letter_date": "January 18, 2023"}, 2023, "letter 2023-01"),
    ({"bulletin_id": "2024-001", "effective_from": "2024-01-08"}, 2024, "bulletin 2024-001"),
    ({"formulary_year": 2025, "effective_from": "2025-01-01"}, 2025, "plan year 2025"),
    ({"case_id": "APL-1", "filed_date": "2025-09-08", "decided_date": "2025-09-20"}, 2025, "decided date 2025-09-20"),
    ({"call_id": "C1", "call_date": "2026-08-31"}, 2026, "call date 2026-08-31"),
    ({}, 2026, "2026 edition"),
])
def test_document_version_reads_governed_metadata(record, year, label):
    c = _candidate(0.9)
    c.record, c.year = record, year
    assert payload.document_version(c) == label


def test_one_point_zero_shape_still_validates():
    built = payload.build("q", IN_SCOPE, [_candidate(0.9)])
    built["spec_version"] = "1.0.0"
    del built["chunks"][0]["source"]["doc_type"]
    jsonschema.validate(built, SCHEMA)
