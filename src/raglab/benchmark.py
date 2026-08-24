"""HNSW-vs-exact benchmark. Exact scan (index disabled) is ground truth;
recall@k measures how much of the true top-k the index returns at each
ef_search setting. Queries are sampled real chunk embeddings — standard
ANN-benchmark methodology."""

import statistics
import time
from dataclasses import dataclass

import psycopg

EF_VALUES = (10, 20, 40, 80, 120, 200)


@dataclass
class BenchmarkRow:
    ef_search: int
    recall: float
    median_ms: float


@dataclass
class BenchmarkResult:
    n_vectors: int
    n_queries: int
    k: int
    exact_median_ms: float
    rows: list[BenchmarkRow]


def _topk(conn: psycopg.Connection, vec: str, k: int) -> tuple[list[float], float]:
    """Returns the top-k distances and the elapsed milliseconds."""
    t0 = time.perf_counter()
    rows = conn.execute(
        "SELECT embedding <=> %s::vector AS d FROM chunks "
        "WHERE embedding IS NOT NULL ORDER BY d LIMIT %s",
        (vec, k),
    ).fetchall()
    return [r[0] for r in rows], (time.perf_counter() - t0) * 1000


def run(conn: psycopg.Connection, n_queries: int = 100, k: int = 10) -> BenchmarkResult:
    n_vectors = conn.execute(
        "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    queries = [
        row[0]
        for row in conn.execute(
            "SELECT embedding::text FROM chunks WHERE embedding IS NOT NULL "
            "ORDER BY random() LIMIT %s",
            (n_queries,),
        ).fetchall()
    ]

    # Ground truth under a transaction-local planner override. Recall is
    # distance-based: the corpus contains exact-duplicate embeddings
    # (cross-year boilerplate), so tied neighbors with different ids are
    # equally correct and id-based recall would undercount.
    truth_kth: list[float] = []
    exact_times: list[float] = []
    conn.execute("SET LOCAL enable_indexscan = off")
    for vec in queries:
        dists, ms = _topk(conn, vec, k)
        truth_kth.append(dists[-1])
        exact_times.append(ms)
    conn.rollback()  # clears SET LOCAL

    eps = 1e-9
    rows = []
    for ef in EF_VALUES:
        # Pin the planner to the index: at higher ef_search its cost estimate
        # exceeds seq-scan cost and it silently reverts to exact scan,
        # which would corrupt the measurement.
        conn.execute("SET LOCAL enable_seqscan = off")
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef),))
        recalls, times = [], []
        for vec, kth in zip(queries, truth_kth, strict=True):
            dists, ms = _topk(conn, vec, k)
            recalls.append(sum(1 for d in dists if d <= kth + eps) / k)
            times.append(ms)
        rows.append(
            BenchmarkRow(
                ef_search=ef,
                recall=statistics.mean(recalls),
                median_ms=statistics.median(times),
            )
        )
        conn.rollback()
    return BenchmarkResult(
        n_vectors=n_vectors,
        n_queries=n_queries,
        k=k,
        exact_median_ms=statistics.median(exact_times),
        rows=rows,
    )


def markdown_table(result: BenchmarkResult) -> str:
    lines = [
        f"Corpus: {result.n_vectors} vectors (1536d) · {result.n_queries} queries · "
        f"recall@{result.k} vs exact scan (median {result.exact_median_ms:.1f} ms)",
        "",
        "| ef_search | recall@10 | median latency (ms) |",
        "|---:|---:|---:|",
    ]
    for row in result.rows:
        lines.append(f"| {row.ef_search} | {row.recall:.3f} | {row.median_ms:.1f} |")
    return "\n".join(lines)
