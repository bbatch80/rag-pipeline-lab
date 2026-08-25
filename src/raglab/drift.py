"""Embedding-drift check: distribution statistics over a stable chunk sample,
compared against the previous drift run. Detects silent embedding-space shift
(model updates, re-embed anomalies) before it degrades retrieval.

Sample identity is (source_path, chunk_index) — stable across rebuilds even
though chunk ids are not."""

import json
import math
from dataclasses import dataclass

import psycopg

from raglab import eval_retrieval

SAMPLE_SIZE = 500
CENTROID_SHIFT_ALERT = 0.02  # cosine distance between run centroids


@dataclass
class DriftResult:
    run_id: int
    sample_size: int
    norm_mean: float
    centroid_shift: float | None  # None on first run
    alert: bool


def run(conn: psycopg.Connection) -> DriftResult:
    conn.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())

    rows = conn.execute(
        """
        SELECT c.embedding::text FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY md5(d.source_path || ':' || c.chunk_index)
        LIMIT %s
        """,
        (SAMPLE_SIZE,),
    ).fetchall()
    vectors = [[float(x) for x in r[0].strip("[]").split(",")] for r in rows]
    if not vectors:
        raise RuntimeError("no embedded chunks to sample")

    dim = len(vectors[0])
    centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    norm_mean = sum(math.sqrt(sum(x * x for x in v)) for v in vectors) / len(vectors)

    previous = conn.execute(
        """
        SELECT s.detail->'centroid' FROM eval_scores s
        JOIN eval_runs r ON r.id = s.run_id
        WHERE r.kind = 'drift' AND s.metric = 'centroid'
        ORDER BY s.run_id DESC LIMIT 1
        """
    ).fetchone()

    shift = None
    if previous and previous[0]:
        prev = previous[0]
        dot = sum(a * b for a, b in zip(centroid, prev))
        na = math.sqrt(sum(a * a for a in centroid))
        nb = math.sqrt(sum(b * b for b in prev))
        shift = 1.0 - dot / (na * nb) if na and nb else None

    run_id = conn.execute(
        "INSERT INTO eval_runs (kind, config_label, git_sha) "
        "VALUES ('drift', 'scheduled', %s) RETURNING id",
        (eval_retrieval._git_sha(),),
    ).fetchone()[0]
    alert = shift is not None and shift > CENTROID_SHIFT_ALERT
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) "
            "VALUES (%s, '-', 'drift', %s, %s, %s)",
            [
                (run_id, "norm_mean", norm_mean, "{}"),
                (run_id, "centroid_shift", shift if shift is not None else 0.0,
                 json.dumps({"first_run": shift is None})),
                (run_id, "centroid", 0.0, json.dumps({"centroid": centroid})),
            ],
        )
    conn.commit()
    return DriftResult(
        run_id=run_id, sample_size=len(vectors), norm_mean=norm_mean,
        centroid_shift=shift, alert=alert,
    )
