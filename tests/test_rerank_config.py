"""One reranker ships; its abstention threshold travels with it."""

from raglab import rerank


def test_single_pinned_reranker():
    assert list(rerank.RERANKERS) == ["bge-base"]
    assert rerank.RERANKER == "bge-base"
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
