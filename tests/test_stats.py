"""Confidence intervals and paired run diffs."""

import pytest


from raglab import eval_retrieval, stats

pytestmark = pytest.mark.readonly


def test_wilson_matches_known_values():
    low, high = stats.wilson(28, 29)          # v1's hit@5 0.966 on 29 answerable questions
    assert 0.82 < low < 0.84 and 0.99 < high <= 1.0
    assert stats.wilson(0, 10)[0] == 0.0 and stats.wilson(10, 10)[1] == 1.0
    assert stats.wilson(0, 0) == (0.0, 1.0)
    low, high = stats.wilson(50, 100)
    assert abs(low - 0.404) < 0.01 and abs(high - 0.596) < 0.01


def test_paired_diff_names_the_questions():
    before = {"F1": 1.0, "F7": 1.0, "Y2": 0.0, "T3": 1.0}
    after = {"F1": 1.0, "F7": 0.0, "Y2": 1.0, "L9": 1.0}
    d = stats.paired_diff(before, after)
    assert d == {"gained": ["Y2"], "lost": ["F7"], "only_before": ["T3"],
                 "only_after": ["L9"], "n": 3}


def test_summary_carries_intervals_and_n():
    scores = [(f"q{i}", "factual", "hit@5", 1.0 if i < 9 else 0.0, {"source": "brochure"})
              for i in range(10)]
    scores += [(f"q{i}", "factual", "precision@5", 0.4, {"source": "brochure"}) for i in range(10)]
    result = eval_retrieval._summarize(0, scores)
    cat = result.by_category["factual"]
    assert cat["hit@5"] == 0.9 and cat["n"] == 10
    low, high = cat["hit@5_ci"]
    assert low < 0.9 < high and "precision@5_ci" not in cat
    assert result.overall_ci["hit@5"] == cat["hit@5_ci"]
    assert result.by_source["brochure"]["hit@5_ci"] == cat["hit@5_ci"]


def test_diff_runs_reads_two_runs(db):
    db.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())
    a, b = (
        db.execute("INSERT INTO eval_runs (kind, config_label) VALUES ('retrieval', %s) RETURNING id", (lbl,)).fetchone()[0]
        for lbl in ("a", "b")
    )
    rows = [(a, "F7", 1.0), (a, "Y2", 0.0), (b, "F7", 0.0), (b, "Y2", 1.0)]
    for run, q, v in rows:
        db.execute("INSERT INTO eval_scores (run_id, question_id, category, metric, value) "
                   "VALUES (%s, %s, 'x', 'hit@5', %s)", (run, q, v))
    d = eval_retrieval.diff_runs(db, a, b)
    assert d == {"hit@5": {"gained": ["Y2"], "lost": ["F7"], "only_before": [], "only_after": [], "n": 2}}
