"""Payload spec contract: abstention decided upstream, carried in status."""

from raglab import payload
from raglab.retrieval import Candidate
from raglab.router import Route


def _candidate(score, **kw):
    base = dict(
        chunk_id=1, content="text", section="s", doc_title="d", source_path="p",
        plan_code="71-006", year=2026, acl_tag="public", pages=[14],
        vector_rank=1, text_rank=None, rrf_score=0.01, rerank_score=score,
    )
    base.update(kw)
    return Candidate(**base)


def test_out_of_scope_payload_carries_boundary_response():
    route = Route(scope="out_of_year", boundary_response="Corpus covers 2021-2026.")
    built = payload.build("q", route, [])
    assert built["status"] == "out_of_scope"
    assert built["boundary_response"] == "Corpus covers 2021-2026."
    assert built["chunks"] == []


def test_low_confidence_payload_signals_insufficient_evidence():
    route = Route(scope="in_scope", years=(2026,))
    built = payload.build("q", route, [_candidate(0.12)])
    assert built["status"] == "insufficient_evidence"


def test_ok_payload_carries_provenance_and_scores():
    route = Route(scope="in_scope", years=(2026,), plan_codes=("71-006",))
    built = payload.build("q", route, [_candidate(0.91)])
    assert built["status"] == "ok"
    assert built["confidence"] == 0.91
    chunk = built["chunks"][0]
    assert chunk["source"]["plan_code"] == "71-006"
    assert chunk["source"]["pages"] == [14]
    assert chunk["scores"]["rerank"] == 0.91
    assert built["router"]["years"] == [2026]


def test_payload_caps_chunk_count():
    route = Route(scope="in_scope", years=(2026,))
    cands = [_candidate(0.9, chunk_id=i) for i in range(20)]
    built = payload.build("q", route, cands, max_chunks=8)
    assert len(built["chunks"]) == 8
