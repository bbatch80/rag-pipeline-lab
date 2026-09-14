"""The eval scores questions on worker threads. What must hold: results
come back in golden order (a parallel run stores what a sequential run
would), every worker scores on its own connection, the reranker's cache
connection is per thread, and prediction itself is serialized."""

import threading
import time

from raglab import eval_retrieval, rerank


class _Conn:
    _n = 0

    def __init__(self):
        _Conn._n += 1
        self.id = _Conn._n
        self.closed = False

    def close(self):
        self.closed = True


def test_parallel_scoring_keeps_golden_order_and_one_connection_per_worker(monkeypatch):
    import raglab.db as raglab_db

    conns = []
    monkeypatch.setattr(raglab_db, "connect", lambda: conns.append(_Conn()) or conns[-1])  # imported lazily by the pool
    monkeypatch.setattr(rerank, "_get_model", lambda: None)
    seen = []
    lock = threading.Lock()

    def fake_score(conn, item):
        time.sleep(0.01 * (5 - int(item["id"][-1])))  # later items finish first
        with lock:
            seen.append((item["id"], conn.id, threading.current_thread().name))
        return [("hit@5", 1.0, {"item": item["id"]})]

    monkeypatch.setattr(eval_retrieval, "score_item", fake_score)
    items = [{"id": f"q-{i}", "category": "factual"} for i in range(1, 5)]
    rows = eval_retrieval._score_items_parallel(items, workers=3)
    assert [r[0][2]["item"] for r in rows] == ["q-1", "q-2", "q-3", "q-4"]  # golden order, whatever finished first
    assert len(conns) == 3 and all(c.closed for c in conns)               # one connection per worker, all closed
    assert {conn_id for _, conn_id, _ in seen} <= {c.id for c in conns}
    assert all(name.startswith("eval") for _, _, name in seen)


def test_sequential_path_is_unchanged_for_one_worker(monkeypatch):
    """workers=1 never touches the pool: the caller's connection scores everything."""
    monkeypatch.setattr(eval_retrieval, "_score_items_parallel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pool used")))
    called = []
    monkeypatch.setattr(eval_retrieval, "score_item", lambda conn, item: called.append(item["id"]) or [])
    monkeypatch.setattr(eval_retrieval.ablation, "load_golden", lambda: [{"id": "a-1", "category": "x"}, {"id": "b-1", "category": "y"}])
    monkeypatch.setattr(eval_retrieval, "_item_verdict_rows", lambda scores: [])
    monkeypatch.setattr(eval_retrieval, "expected_source", lambda item, registry: "none")
    monkeypatch.setattr(eval_retrieval, "_summarize", lambda run_id, scores: eval_retrieval.RetrievalEvalResult(run_id=run_id))
    monkeypatch.setattr(eval_retrieval, "corpus_hash", lambda conn: "h")
    monkeypatch.setattr(eval_retrieval.sources, "load", lambda conn: None)

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def executemany(self, *a, **k):
            pass

    class _DB:
        def execute(self, sql, *a, **k):
            class _R:
                def fetchone(self_inner):
                    return (7,) if "INSERT INTO eval_runs" in sql else (None,)  # max(id) of no previous run is a NULL row
            return _R()

        def cursor(self):
            return _Cur()

        def commit(self):
            pass

    monkeypatch.setattr(eval_retrieval, "EVAL_SCHEMA_PATH", type("P", (), {"read_text": lambda self: "SELECT 1"})())
    result = eval_retrieval.run(_DB(), config_label="t", workers=1, categories=("x",))
    assert called == ["a-1"] and result.run_id == 7


def test_rerank_cache_connection_is_per_thread_and_prediction_is_serialized(monkeypatch):
    import raglab.db as raglab_db

    made = []
    monkeypatch.setattr(raglab_db, "connect", lambda: made.append(_Conn()) or made[-1])  # _cache_connection imports raglab.db lazily
    monkeypatch.setattr(rerank, "_cache_local", threading.local())
    ids = {}

    def grab(name):
        ids[name] = rerank._cache_connection().id

    threads = [threading.Thread(target=grab, args=(n,)) for n in ("t1", "t2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ids["t1"] != ids["t2"] and len(made) == 2

    active, peak = {"n": 0}, {"n": 0}
    guard = threading.Lock()

    class _Model:
        def predict(self, pairs):
            with guard:
                active["n"] += 1
                peak["n"] = max(peak["n"], active["n"])
            time.sleep(0.05)
            with guard:
                active["n"] -= 1
            return [0.5] * len(pairs)

    monkeypatch.setattr(rerank, "RERANK_DTYPE", "fp32")
    workers = [threading.Thread(target=lambda: rerank._predict(_Model(), [("q", "t")])) for _ in range(4)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    assert peak["n"] == 1  # one prediction at a time
