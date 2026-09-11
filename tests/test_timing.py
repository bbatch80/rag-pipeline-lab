"""Per-stage latency: measured on every query, stored with the disclosure row,
summarized as p50/p95 per eval run."""

import time

from raglab import eval_retrieval, timing
from raglab.pipeline import run_query


def test_stopwatch_stages_sum_within_total():
    watch = timing.Stopwatch()
    with watch.stage("a"):
        time.sleep(0.01)
    with watch.stage("b"):
        time.sleep(0.01)
    snap = watch.snapshot()
    assert set(snap) >= {"a", "b", "total", "host"}
    assert snap["a"] >= 9 and snap["b"] >= 9
    assert snap["total"] >= snap["a"] + snap["b"]


def test_percentile_nearest_rank():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert timing.percentile(vals, 50) == 50.0
    assert timing.percentile(vals, 95) == 100.0
    assert timing.percentile([], 95) == 0.0


def test_run_query_stamps_timings_on_the_disclosure_row(db, monkeypatch):
    monkeypatch.setattr("raglab.retrieval.embed_query", lambda text: "[" + ",".join(["0.5"] * 1536) + "]")

    class _FakeModel:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    import raglab.rerank as rr
    monkeypatch.setattr(rr, "_model", _FakeModel())

    class _NoCommit:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *a, **k):
            return self._conn.execute(*a, **k)

        def commit(self):
            pass

    built = run_query(_NoCommit(db), "what is the deductible", persona="public")
    row = db.execute(
        "SELECT timings FROM disclosure_log WHERE payload_id = %s", (built["payload_id"],)
    ).fetchone()
    assert row and row[0], "timings stored with the disclosure row"
    timings = row[0]
    for stage in ("route", "embed", "search", "rerank", "payload", "disclose", "total"):
        assert stage in timings, stage
    assert timings["total"] >= timings["search"] + timings["rerank"]
    assert "timings" in built and built["timings"]["total"] >= 0, "spec 1.1.0 carries per-stage timings (D5)"


def test_summary_reports_latency_percentiles():
    scores = [(f"q{i}", "factual", "latency_total", float(100 + i * 10), {"host": "t"}) for i in range(20)]
    scores += [(f"q{i}", "factual", "latency_search", 5.0, {"host": "t"}) for i in range(20)]
    result = eval_retrieval._summarize(0, scores)
    assert result.latency["total"]["n"] == 20
    assert result.latency["total"]["p50"] == 190.0 and result.latency["total"]["p95"] == 280.0
    assert result.latency["search"] == {"p50": 5.0, "p95": 5.0, "n": 20}
