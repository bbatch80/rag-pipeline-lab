"""Year-over-year source coverage is ratchet-gated at the reproduced baseline."""

from raglab import eval_retrieval


def _scores(coverage_values):
    return [(f"Y{i}", "yoy", "source_coverage", v, {}) for i, v in enumerate(coverage_values)] + [
        (f"Y{i}", "yoy", "hit@5", 1.0, {}) for i, _ in enumerate(coverage_values)
    ]


def test_ratchet_allows_one_half_miss():
    result = eval_retrieval._summarize(0, _scores([1.0] * 7 + [0.5]))  # 0.9375 >= 0.9
    assert not [f for f in result.failures if "yoy" in f]


def test_ratchet_fires_at_two_half_misses():
    result = eval_retrieval._summarize(0, _scores([1.0] * 6 + [0.5, 0.5]))  # 0.875 < 0.9
    assert any("yoy source_coverage 0.875 < ratchet" in f for f in result.failures)
