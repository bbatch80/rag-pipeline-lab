"""The discrimination check breaks both retrieval arms, not just the vector."""

import pytest

from raglab import eval_retrieval, retrieval

pytestmark = pytest.mark.slow


def test_sabotage_breaks_both_arms(db, monkeypatch):
    seen = []

    def fake_search(conn, text, vector, route, **kw):
        seen.append((text, vector[:12]))
        return []

    monkeypatch.setattr(retrieval, "search", fake_search)
    monkeypatch.setattr(retrieval, "embed_query", lambda t: "[REAL]")
    import raglab.rerank as rr
    monkeypatch.setattr(rr, "rerank", lambda *a, **k: [])
    class _NoCommit:  # eval_retrieval.run commits; the fixture forbids it
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *a, **k):
            return self._conn.execute(*a, **k)

        def cursor(self, *a, **k):
            return self._conn.cursor(*a, **k)

        def commit(self):
            pass

        def transaction(self, *a, **k):  # savepoints are fine; only commit is forbidden
            return self._conn.transaction(*a, **k)

    eval_retrieval.run(_NoCommit(db), config_label="t", sabotage=True)
    assert seen, "answerable questions were searched"
    assert all(text == "zzqx zzqv zzqw" for text, _ in seen), "lexical arm sabotaged"
    assert all(vec.startswith("[0.01,0.01") for _, vec in seen), "vector arm sabotaged"
