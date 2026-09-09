"""Change questions are searched and reranked per year on their subject."""

from raglab import rerank, router
from raglab.retrieval import Candidate


def test_year_neutral_strips_change_language_and_other_years():
    q = "How did the High Option specialist copay change from 2025 to 2026?"
    assert router.year_neutral(q, 2025) == "the High Option specialist copay 2025"
    assert router.year_neutral(q, 2026) == "the High Option specialist copay 2026"
    assert "in-network" in router.year_neutral("How did the GEHA HDHP in-network deductible change from 2025 to 2026?", 2025)
    q2 = "What was the GEHA Benefit Plan in-network deductible in 2021, and how does it compare to 2026?"
    out = router.year_neutral(q2, 2021)
    assert "2026" not in out and "compare" not in out and out.endswith("2021")
    assert "deductible" in out
    assert router.year_queries("What is the deductible?", (2026,)) == {}


def test_rerank_scores_each_year_against_its_own_query(monkeypatch):
    seen = []

    class _Model:
        def predict(self, pairs):
            seen.extend(pairs)
            return [0.5] * len(pairs)

    monkeypatch.setattr(rerank, "_model", _Model())
    cands = [Candidate(chunk_id=i, content=f"c{i}", section="", doc_title="", source_path="", plan_code=None,
                       year=y, acl_tag="public", pages=[], vector_rank=1, text_rank=1, rrf_score=0.1)
             for i, y in enumerate((2025, 2026))]
    rerank.rerank("How did the deductible change from 2025 to 2026?", cands, stratify_years=(2025, 2026))
    assert [q for q, _ in seen] == ["the deductible 2025", "How did the deductible change from 2025 to 2026?"]
