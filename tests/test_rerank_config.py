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


def test_bf16_precision_is_in_the_cache_key_and_wraps_prediction(monkeypatch):
    """The precision switch: off by default; on, every prediction runs under
    CPU bf16 autocast and the cache key carries it so fp32 and bf16 scores
    never mix."""
    import raglab.rerank as rr

    monkeypatch.setattr(rr, "RERANK_DTYPE", "fp32")
    monkeypatch.setattr(rr, "_model_key", "m@rev")
    assert rr.model_key().endswith("m@rev") or "+bf16" not in rr.model_key()
    monkeypatch.setattr(rr, "RERANK_DTYPE", "bf16")
    assert rr.model_key().endswith("+bf16")
    import torch

    seen = {}

    class _Probe:
        def predict(self, pairs):
            import torch
            seen["autocast"] = torch.is_autocast_enabled("cpu")
            seen["dtype"] = torch.get_autocast_dtype("cpu") if seen["autocast"] else None
            return [0.5] * len(pairs)

    assert rr._predict(_Probe(), [("q", "t"), ("q", "u")]) == [0.5, 0.5]
    assert seen == {"autocast": True, "dtype": torch.bfloat16}
    monkeypatch.setattr(rr, "RERANK_DTYPE", "fp32")
    rr._predict(_Probe(), [("q", "t")])
    assert seen["autocast"] is False


def test_cover_all_dedupes_plans_without_guaranteeing_seats(monkeypatch):
    """Nothing named → every plan covered, but no plan is owed a seat: score
    order with at most two chunks per plan, and plan-less sources compete
    as they are (run 667: guaranteed lanes took a bulletin's seats)."""
    from raglab import rerank
    from raglab.retrieval import Candidate

    def cand(cid, plan, score, dt="brochure"):
        return Candidate(chunk_id=cid, content="", section="", doc_title=f"{dt}-{cid}", source_path="", plan_code=plan,
                         year=2026, acl_tag="public", pages=[], vector_rank=1, text_rank=1, rrf_score=0.0, rerank_score=score, doc_type=dt)

    pool = [cand(1, "71-006", 0.9), cand(2, "71-006", 0.85), cand(3, "71-006", 0.8), cand(4, "71-014", 0.7),
            cand(5, None, 0.75, "bulletin"), cand(6, "71-018", 0.1), cand(7, "71-021", 0.05), cand(8, "71-026", 0.04)]

    import raglab.rerank as rr
    monkey_scores = {c.chunk_id: c.rerank_score for c in pool}
    monkeypatch.setattr(rr, "_get_model", lambda: type("M", (), {"predict": staticmethod(lambda pairs: [0.0] * len(pairs))})())
    monkeypatch.setattr(rr, "score_pairs", lambda model, pairs, conn=None: [monkey_scores[int(t.split("#")[1])] if "#" in t else 0.0 for _, t in pairs])
    for c in pool:
        c.index_text = f"chunk#{c.chunk_id}"
    plans = ("71-006", "71-014", "71-018", "71-021", "71-026")
    seated = rr.rerank("q", list(pool), top_n=5, stratify_years=(2026,), stratify_plans=plans, plan_seats=False)
    assert [c.chunk_id for c in seated] == [1, 2, 5, 4, 6]   # two of 71-006 max; the bulletin keeps its score place; no seat for 0.05
    guaranteed = rr.rerank("q", list(pool), top_n=5, stratify_years=(2026,), stratify_plans=plans, plan_seats=True)
    plans_seated = {c.plan_code for c in guaranteed} - {None}
    assert len(plans_seated) >= 4 and any(c.doc_type == "bulletin" for c in guaranteed)  # the program/option rule still seats the lanes (5 seats: 4 lanes + the bulletin that outscores them)
