"""Retrieval contract tests: RRF math, rare-lexeme query building,
relevance matching, rerank/abstention mechanics (fake model — no weights)."""

import pytest

from raglab import ablation, rerank
from raglab.retrieval import Candidate, RRF_K, _lexical_query, search
from raglab.router import Route


def _candidate(**kw):
    base = dict(
        chunk_id=1, content="x", section="", doc_title="t", source_path="p",
        plan_code=None, year=2026, acl_tag="public", pages=[],
        vector_rank=None, text_rank=None, rrf_score=0.0,
    )
    base.update(kw)
    return Candidate(**base)


@pytest.fixture
def tiny_corpus(db):
    doc_id = db.execute(
        "INSERT INTO documents (source_path, title, content_hash) "
        "VALUES ('t/rrf.pdf', 'RRF Fixture', 'h') RETURNING id"
    ).fetchone()[0]
    # Three chunks with hand-built vectors: chunk A nearest to the query
    # vector, chunk C matches the rare term 'zephyrite'. Vectors vary in
    # DIRECTION (constant vectors are all parallel — cosine distance 0).
    def vec(x):
        rest = ",".join(["0.0"] * 1534)
        return f"[1.0,{x:.3f},{rest}]"

    contents = [
        ("alpha benefit text", vec(0.100)),          # A: vector rank 1
        ("beta benefit text", vec(0.300)),           # B: vector rank 2
        ("gamma zephyrite identifier", vec(0.900)),  # C: vector rank 3
    ]
    for i, (content, v) in enumerate(contents):
        db.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, year, embedding) "
            "VALUES (%s, %s, %s, 2026, %s::vector)",
            (doc_id, i, content, v),
        )
    db.execute("DROP TABLE IF EXISTS lexeme_df")
    db.execute(
        "CREATE TABLE lexeme_df AS SELECT word, ndoc "
        "FROM ts_stat('SELECT tsv FROM chunks')"
    )
    return db, vec


def test_rrf_math_matches_hand_computation(tiny_corpus):
    db, vec = tiny_corpus
    route = Route(scope="in_scope", years=(2026,))
    results = search(db, "zephyrite", vec(0.100), route)
    by_id = {c.content.split()[0]: c for c in results}

    # A: vector rank 1, no text match -> 1/(60+1)
    assert by_id["alpha"].rrf_score == pytest.approx(1 / (RRF_K + 1))
    # C: text rank 1 (only 'zephyrite' match) + vector rank 3
    assert by_id["gamma"].text_rank == 1
    assert by_id["gamma"].rrf_score == pytest.approx(
        1 / (RRF_K + 3) + 1 / (RRF_K + 1)
    )
    # Fusion promotes C (both arms) above A (one arm, better rank).
    assert results[0].content.startswith("gamma")


def test_lexical_query_keeps_only_rare_terms(tiny_corpus):
    db, _ = tiny_corpus
    # 'benefit' appears in 2 of 3 chunks (common at this scale is >1%);
    # 'zephyrite' in 1. Both are candidates; the rare filter keeps zephyrite.
    q = _lexical_query(db, "benefit zephyrite")
    assert "zephyrit" in q  # stemmed lexeme
    assert "benefit" not in q


def test_lexical_query_falls_back_when_no_rare_terms(tiny_corpus):
    db, _ = tiny_corpus
    q = _lexical_query(db, "benefit text")
    assert "benefit" in q and "text" in q, "no rare terms -> use all lexemes"


def test_is_relevant_brochure_page_offset():
    c = _candidate(plan_code="71-006", year=2026, pages=[16])
    assert ablation.is_relevant(c, [{"plan_code": "71-006", "year": 2026,
                                     "printed_pages": [14]}])
    assert not ablation.is_relevant(c, [{"plan_code": "71-006", "year": 2025,
                                         "printed_pages": [14]}])
    assert not ablation.is_relevant(c, [{"plan_code": "71-014", "year": 2026,
                                         "printed_pages": [14]}])


def test_is_relevant_internal_suffix():
    c = _candidate(source_path="data/internal/kb/kb_mail_order.md")
    assert ablation.is_relevant(c, [{"internal": "kb/kb_mail_order.md"}])
    assert not ablation.is_relevant(c, [{"internal": "kb/kb_other.md"}])


class _FakeModel:
    def predict(self, pairs):
        # Score by presence of 'answer' in the chunk text.
        return [0.9 if "answer" in chunk else 0.1 for _, chunk in pairs]


def test_rerank_orders_and_verdicts(monkeypatch):
    monkeypatch.setattr(rerank, "_model", _FakeModel())
    cands = [
        _candidate(chunk_id=1, content="noise text"),
        _candidate(chunk_id=2, content="the answer text"),
    ]
    ordered = rerank.rerank("q", cands)
    assert ordered[0].chunk_id == 2
    abstain, best = rerank.abstention_verdict(ordered)
    assert not abstain and best == pytest.approx(0.9)

    ordered_noise = rerank.rerank("q", [_candidate(content="noise")])
    abstain, best = rerank.abstention_verdict(ordered_noise)
    assert abstain, "best score below threshold must abstain"
