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


def test_allow_wall_is_judged_on_the_documents_when_only_the_warehouse_is_missing():
    """CI has no warehouse: a two-leg plan whose case row could not run reads
    insufficient although the letter was found. The allow check judges the
    documents and records the warehouse half as skipped (PR #57's CI gate)."""
    from raglab import eval_retrieval as er

    ci_shaped = {"status": "insufficient_evidence", "missing": ["appeal_case"],
                 "warehouse_results": [{"leg": "appeal_case", "status": "not_executed", "reason": "warehouse unavailable: KeyError"}],
                 "sub_results": [{"leg": "documents", "status": "ok", "chunk_indexes": [0, 1]}]}
    assert er._only_the_warehouse_is_missing(ci_shaped)
    assert not er._only_the_warehouse_is_missing({**ci_shaped, "sub_results": [{"leg": "documents", "status": "insufficient_evidence"}]})
    assert not er._only_the_warehouse_is_missing({**ci_shaped, "missing": ["appeal_case", "documents"]})
    assert not er._only_the_warehouse_is_missing({**ci_shaped, "warehouse_results": [{"leg": "appeal_case", "status": "not_executed", "reason": "case_id required but not in context"}]})
    assert not er._only_the_warehouse_is_missing({**ci_shaped, "status": "ok"})


def test_an_item_that_opens_a_case_is_unverifiable_where_the_case_cannot_resolve(monkeypatch):
    """CI has no population: a case cannot resolve, so neither the record
    filter nor the case-file rule fires. The item's document checks are
    recorded as skipped there, never as failures (PR #73's CI gate)."""
    from raglab import checks, eval_retrieval as er

    item = {"id": "appeal-17", "category": "appeal", "persona": "appeals", "module": "appeals_workbench", "case_id": "APL-4935714",
            "question": "What policy rules applied to this case?", "sources": [{"internal": "appeals/appeal_0018.md"}],
            "expected_legs": ["appeal_case", "doc_probe:appeal"], "expect_status": "ok"}
    ci_payload = {"status": "insufficient_evidence", "record_context": {}, "chunks": [], "sub_results": [], "missing": ["case"],
                  "warehouse_results": [{"leg": "case", "query_name": "appeal_case", "status": "not_executed", "reason": "warehouse unavailable: KeyError"}],
                  "plan": {"legs": [{"name": "case", "kind": "member_query", "query_name": "appeal_case"}], "enforced": []}, "router": {}}
    monkeypatch.setattr(er, "_compose", lambda conn, it, q, persona=None: ci_payload)
    monkeypatch.setattr(checks, "_warehouse_unavailable", lambda payload: True)
    rows = er.score_item(None, item)
    metrics = {m for m, _, _ in rows}
    assert not any(m == "hit@5" or m.startswith("check_") for m in metrics), metrics   # nothing judged; health and latency are reported, not gated
    assert any(d.get("reason") == "case context unavailable" for _, _, d in rows)
    local_payload = {**ci_payload, "record_context": {"case_id": "APL-4935714"}}
    monkeypatch.setattr(er, "_compose", lambda conn, it, q, persona=None: local_payload)
    monkeypatch.setattr(checks, "_warehouse_unavailable", lambda payload: False)
    rows = er.score_item(None, item)
    assert any(m == "hit@5" for m, _, _ in rows)   # with the case in context the checks are judged as usual
