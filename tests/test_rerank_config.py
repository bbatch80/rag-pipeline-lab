"""One reranker ships; its abstention threshold travels with it."""

from raglab import rerank


def test_single_pinned_reranker():
    """bge-base ships; every other entry is a bake-off candidate (carries
    `size_gb`, loaded only by `raglab rerank-bakeoff`)."""
    assert rerank.RERANKER == "bge-base"
    assert "size_gb" not in rerank.RERANKERS["bge-base"]
    assert all("size_gb" in spec for key, spec in rerank.RERANKERS.items() if key != "bge-base")
    assert rerank.MODEL_NAME == "BAAI/bge-reranker-base"
    assert rerank.ABSTAIN_THRESHOLD == rerank.RERANKERS["bge-base"]["threshold"] == 0.5


def test_abstention_is_per_source():
    """A record clears its own (lower) threshold; prose keeps 0.5; nothing
    clearing anything abstains."""
    from raglab.retrieval import Candidate

    def cand(doc_type, score):
        c = Candidate(chunk_id=0, content="", section="", doc_title="", source_path="",
                      plan_code=None, year=2026, acl_tag="public", pages=[], vector_rank=None,
                      text_rank=None, rrf_score=0.0, doc_type=doc_type)
        c.rerank_score = score
        return c

    assert rerank.threshold_for("call_note") == 0.1 and rerank.threshold_for("brochure") == 0.5
    assert rerank.abstention_verdict([cand("brochure", 0.3), cand("call_note", 0.12)]) == (False, 0.3)
    assert rerank.abstention_verdict([cand("brochure", 0.3), cand("call_note", 0.05)]) == (True, 0.3)
    assert rerank.abstention_verdict([cand("call_note", 0.6)]) == (False, 0.6)
    assert rerank.abstention_verdict([]) == (True, 0.0)


def test_rerank_cache_memoizes_on_model_query_and_text(db, monkeypatch):
    """Hits are read, misses scored once in a batch and stored; any change to
    the query text, the chunk text, or the model key misses."""
    from raglab.eval_retrieval import EVAL_SCHEMA_PATH

    db.execute(EVAL_SCHEMA_PATH.read_text())
    calls = []

    class Fake:
        def predict(self, pairs):
            calls.append(list(pairs)); return [0.5 + 0.01 * i for i, _ in enumerate(pairs)]

    monkeypatch.setattr(rerank, "RERANK_CACHE", True)
    monkeypatch.setattr(rerank, "_cache_connection", lambda: db)
    monkeypatch.setattr(rerank, "_model_key", "fake@rev1")
    rerank.CACHE_STATS.update(hits=0, misses=0)
    pairs = [("q one", "text a"), ("q one", "text b"), ("q two", "text a")]
    first = rerank.score_pairs(Fake(), pairs)
    assert len(calls) == 1 and len(calls[0]) == 3 and rerank.CACHE_STATS["misses"] == 3
    second = rerank.score_pairs(Fake(), pairs + [("q one", "text c")])
    assert second[:3] == first and len(calls) == 2 and calls[1] == [("q one", "text c")]
    assert rerank.CACHE_STATS["hits"] == 3
    monkeypatch.setattr(rerank, "_model_key", "fake@rev2")  # a model swap misses everything
    rerank.score_pairs(Fake(), pairs)
    assert len(calls) == 3 and len(calls[2]) == 3
    monkeypatch.setattr(rerank, "RERANK_CACHE", False)
    rerank.score_pairs(Fake(), pairs)
    assert len(calls) == 4, "cache off: plain predict"
