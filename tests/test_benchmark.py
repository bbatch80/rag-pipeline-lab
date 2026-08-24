"""HNSW-vs-exact agreement spot check on synthetic vectors (CI-safe: no API,
no real corpus needed)."""

import random


def _manifold_vectors(rng, n: int = 200) -> list[str]:
    """Random-walk synthetic vectors. Uniform random points in 1536d are all
    near-equidistant, and isolated clusters become unreachable islands in the
    graph — both make HNSW look broken. Real embeddings live on a connected
    manifold; a random walk is the simplest connected structure with local
    neighborhoods."""
    current = [rng.uniform(-1, 1) for _ in range(1536)]
    out = []
    for _ in range(n):
        current = [v + rng.gauss(0, 0.08) for v in current]
        out.append("[" + ",".join(f"{v:.6f}" for v in current) + "]")
    return out


def test_hnsw_agrees_with_exact_at_operating_point(db):
    rng = random.Random(42)
    doc_id = db.execute(
        "INSERT INTO documents (source_path, title, content_hash) "
        "VALUES ('t/b.pdf', 'B', 'h') RETURNING id"
    ).fetchone()[0]
    vectors = _manifold_vectors(rng)
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
            "VALUES (%s, %s, 'x', %s::vector)",
            [(doc_id, i, v) for i, v in enumerate(vectors)],
        )
    # Bulk-load-then-index: the fixture emptied the table, so a pre-existing
    # index graph would be full of invisible entries and starve the scan.
    # DDL is transactional — the rollback restores any real index.
    db.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    # Parallel index-build workers cannot see this transaction's uncommitted
    # rows and intermittently produce a malformed graph (observed: zero-row
    # scans). Single-process build is required for in-transaction fixtures.
    db.execute("SET LOCAL max_parallel_maintenance_workers = 0")
    db.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    k, inflations = 10, []
    for query in vectors[:10]:
        # SET LOCAL persists for the whole transaction, so both toggles are
        # set explicitly before each query — leftover state from the other
        # branch would silently corrupt the measurement.
        db.execute("SET LOCAL enable_indexscan = off")
        db.execute("SET LOCAL enable_seqscan = on")
        exact = db.execute(
            "SELECT embedding <=> %s::vector AS d FROM chunks "
            "WHERE embedding IS NOT NULL ORDER BY d LIMIT %s",
            (query, k),
        ).fetchall()
        kth = exact[-1][0]

        db.execute("SET LOCAL enable_indexscan = on")
        db.execute("SET LOCAL enable_seqscan = off")
        # ef_search=200 makes the search effectively exhaustive on this
        # 200-vector fixture: the test verifies the index MECHANISM agrees
        # with exact scan, deterministically. Recall at the production
        # operating point is measured on the real corpus by `raglab benchmark`
        # (graph builds are nondeterministic; a tuned-ef threshold on tiny
        # synthetic data flips between runs).
        db.execute("SELECT set_config('hnsw.ef_search', '200', true)")
        approx = db.execute(
            "SELECT embedding <=> %s::vector AS d FROM chunks "
            "WHERE embedding IS NOT NULL ORDER BY d LIMIT %s",
            (query, k),
        ).fetchall()
        # Distance-inflation check, not membership: HNSW graph builds are
        # nondeterministic, so "same ids" flakes on tie-adjacent neighbors.
        # A real mechanism defect (stale graph, wrong operator, starved scan)
        # returns short or wildly-worse results; an unlucky graph returns a
        # neighbor a fraction of a percent farther.
        assert len(approx) == k, "index scan returned a short result set"
        inflations.append(approx[-1][0] / max(kth, 1e-12))

    assert max(inflations) <= 1.05, f"worst distance inflation {max(inflations):.3f}"
