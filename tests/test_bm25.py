"""The lexical arm: BM25 (pg_textsearch); identifier-shaped questions weight
the lexical list up in RRF."""

import pytest

from raglab import retrieval, router


def _vec(x: float) -> str:
    v = [0.0] * 1536
    v[0] = 1.0
    v[1] = x
    return "[" + ",".join(f"{a:.6f}" for a in v) + "]"


@pytest.fixture
def corpus(db):
    doc = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id) "
        "VALUES ('t/b.pdf', 'B', 'h', 1) RETURNING id"
    ).fetchone()[0]
    rows = [
        (doc, 0, "The calendar year deductible for the HDHP is two thousand dollars.", _vec(0.0)),
        (doc, 1, "Member ID GEHA-04821733 enrolled in the HDHP on January first.", _vec(0.5)),
        (doc, 2, "Preventive care is covered in full in network.", _vec(1.0)),
    ]
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, year, doc_type, embedding) "
            "VALUES (%s, %s, %s, 2026, 'brochure', %s::vector)", rows,
        )
    return doc


def test_bm25_arm_ranks_by_term_relevance(corpus, db):
    pool = retrieval.search(db, "deductible HDHP", _vec(1.0), router.Route(scope="in_scope", years=(2026,)))
    by_text = sorted((c for c in pool if c.text_rank), key=lambda c: c.text_rank)
    assert by_text and "deductible" in by_text[0].content
    assert all(c.text_rank is None for c in pool if "Preventive" in c.content)



def test_identifier_shaped_queries_weight_the_lexical_arm():
    assert retrieval.ID_SHAPED.search("member ID GEHA-04821733 deductible")
    assert retrieval.ID_SHAPED.search("claim CLM-2026-000418")
    assert not retrieval.ID_SHAPED.search("what is the deductible for 2026")
    assert not retrieval.ID_SHAPED.search("High Option coinsurance")
    assert not retrieval.ID_SHAPED.search("a 30-day supply"), "short, two digits: not an identifier"


def test_identifier_query_lets_exact_match_win(corpus, db):
    """Vector says chunk 2 (nearest); the identifier is in chunk 1. With the
    lexical list weighted, the exact match tops the fused pool."""
    pool = retrieval.search(db, "GEHA-04821733", _vec(1.0), router.Route(scope="in_scope", years=(2026,)))
    assert "GEHA-04821733" in pool[0].content


def test_each_source_ranks_within_its_own_index(corpus, db):
    """A matching SOP and a matching brochure chunk each get lexical rank 1:
    ranks are per source (own BM25 statistics), scores never compared."""
    sop = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id, acl_tag) "
        "VALUES ('t/s.md', 'S', 'h', 3, 'employee') RETURNING id"
    ).fetchone()[0]
    db.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, year, doc_type, embedding, metadata) "
        "VALUES (%s, 0, 'Procedure: verify the HDHP deductible before adjudication.', 2026, 'sop', %s::vector, %s::jsonb)",
        (sop, _vec(3.0), '{"record": {"effective_from": "2026-01-01", "effective_to": null}}'),  # sops are versioned
    )
    pool = retrieval.search(db, "deductible HDHP", _vec(1.0), router.Route(scope="in_scope", years=(2026,)))
    by_type = {c.doc_type: c.text_rank for c in pool if c.text_rank == 1}
    assert by_type.get("brochure") == 1 and by_type.get("sop") == 1
