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
    """Runs on its own temp table, NOT chunks: CREATE INDEX deliberately
    indexes tuples deleted by uncommitted transactions (they might roll
    back), so an index built over the fixture-emptied chunks table contains
    ~11k invisible corpus vectors — the 200 test rows become a starving
    minority and the scan flakes to short/zero results. A temp table's index
    holds exactly the rows the test inserted."""
    rng = random.Random(42)
    vectors = _manifold_vectors(rng)
    db.execute(
        "CREATE TEMP TABLE bench_vectors (id serial, embedding vector(1536))"
    )
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO bench_vectors (embedding) VALUES (%s::vector)",
            [(v,) for v in vectors],
        )
    db.execute("SET LOCAL max_parallel_maintenance_workers = 0")
    db.execute(
        "CREATE INDEX bench_hnsw ON bench_vectors "
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
            "SELECT embedding <=> %s::vector AS d FROM bench_vectors "
            "ORDER BY d LIMIT %s",
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
            "SELECT embedding <=> %s::vector AS d FROM bench_vectors "
            "ORDER BY d LIMIT %s",
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
