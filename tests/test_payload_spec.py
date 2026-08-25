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
