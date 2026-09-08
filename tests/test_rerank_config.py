"""One reranker ships; its abstention threshold travels with it."""

from raglab import rerank


def test_single_pinned_reranker():
    assert list(rerank.RERANKERS) == ["bge-base"]
    assert rerank.RERANKER == "bge-base"
    assert rerank.MODEL_NAME == "BAAI/bge-reranker-base"
    assert rerank.ABSTAIN_THRESHOLD == rerank.RERANKERS["bge-base"]["threshold"] == 0.5
