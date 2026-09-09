"""Per-source floor in the candidate pool, and per-source eval slices."""

from raglab import eval_retrieval, retrieval, router, sources


def _vec(x: float) -> str:
    v = [0.0] * 1536
    v[0] = 1.0
    v[1] = x
    return "[" + ",".join(f"{a:.6f}" for a in v) + "]"


def test_floor_keeps_a_crowded_out_source_in_the_pool(db, monkeypatch):
    monkeypatch.setattr(retrieval, "POOLS", "global")  # the floor is the global-pool safeguard
    monkeypatch.setattr(retrieval, "SOURCE_FLOOR", 5)
    """60 brochure chunks sit nearer the query than 3 SOP chunks. A flat
    top-50 pool holds brochures only; the floor admits the SOPs' best."""
    brochure = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id) "
        "VALUES ('t/b.pdf', 'B', 'h', 1) RETURNING id"
    ).fetchone()[0]
    sop = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id, acl_tag) "
        "VALUES ('t/s.md', 'S', 'h', 3, 'employee') RETURNING id"
    ).fetchone()[0]
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, year, doc_type, embedding) "
            "VALUES (%s, %s, %s, 2026, %s, %s::vector)",
            [(brochure, i, f"brochure text {i}", "brochure", _vec(0.01 * i)) for i in range(60)]
            + [(sop, i, f"procedure text {i}", "sop", _vec(5.0 + i)) for i in range(3)],
        )
    route = router.Route(scope="in_scope", years=(2026,))
    # The fixture's DELETE leaves the live corpus's ~21k vectors in the HNSW
    # graph as invisible entries, and an approximate scan walking them can
    # miss a few of the 63 visible rows (47/50 seen twice). This test is
    # about the floor, not the index: take the exact path.
    db.execute("SET LOCAL enable_indexscan = off")
    db.execute("SET LOCAL enable_bitmapscan = off")
    pool = retrieval.search(db, "zzzz", _vec(0.0), route)

    by_type = {}
    for c in pool:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
    assert by_type["brochure"] == 50, "the global pool is unchanged"
    assert by_type["sop"] == 3, "the floor admitted every SOP chunk (fewer than the floor exist)"
    assert all(c.floor for c in pool if c.doc_type == "sop")
    assert not any(c.floor for c in pool if c.doc_type == "brochure")


def test_route_sources_narrows_the_search(db):
    doc = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id) "
        "VALUES ('t/b.pdf', 'B', 'h', 1) RETURNING id"
    ).fetchone()[0]
    db.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, year, doc_type, embedding) "
        "VALUES (%s, 0, 'x', 2026, 'brochure', %s::vector)", (doc, _vec(0.0)),
    )
    only_sops = router.Route(scope="in_scope", years=(2026,), sources=("sop",))
    assert retrieval.search(db, "zzzz", _vec(0.0), only_sops) == []
    everything = router.Route(scope="in_scope", years=(2026,))
    assert len(retrieval.search(db, "zzzz", _vec(0.0), everything)) == 1


def test_expected_source_labels(db):
    registry = sources.load(db)
    assert eval_retrieval.expected_source(
        {"sources": [{"plan_code": "71-006", "year": 2026, "printed_pages": [14]}]}, registry
    ) == "brochure"
    assert eval_retrieval.expected_source(
        {"sources": [{"internal": "bulletins/bulletin_2026_001.md"},
                     {"plan_code": "71-006", "year": 2026, "printed_pages": [1]}]}, registry
    ) == "brochure+bulletin"
    assert eval_retrieval.expected_source({"sources": []}, registry) == "none"


def test_floor_off_leaves_the_pool_untouched(db, monkeypatch):
    monkeypatch.setattr(retrieval, "SOURCE_FLOOR", 0)
    doc = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id) "
        "VALUES ('t/b.pdf', 'B', 'h', 1) RETURNING id"
    ).fetchone()[0]
    db.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, year, doc_type, embedding) "
        "VALUES (%s, 0, 'x', 2026, 'brochure', %s::vector)", (doc, _vec(0.0)),
    )
    pool = retrieval.search(db, "zzzz", _vec(0.0), router.Route(scope="in_scope", years=(2026,)))
    assert len(pool) == 1 and not pool[0].floor


def test_summary_reports_by_source():
    scores = [
        ("q1", "factual", "hit@5", 1.0, {"source": "brochure"}),
        ("q2", "factual", "hit@5", 0.0, {"source": "brochure"}),
        ("q3", "internal_factual", "hit@5", 1.0, {"source": "kb"}),
    ]
    result = eval_retrieval._summarize(0, scores)
    brochure, kb = result.by_source["brochure"], result.by_source["kb"]
    assert (brochure["hit@5"], brochure["n"]) == (0.5, 2) and "hit@5_ci" in brochure
    assert (kb["hit@5"], kb["n"]) == (1.0, 1)
