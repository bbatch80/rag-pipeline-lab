"""The CI gate is a ratchet against the stored baseline: every category and
group holds its baseline pass rate, and a guardrail item that passed at the
baseline may not fail — individually, even when its group's rate holds."""

from raglab import eval_retrieval

GROUP_WORK = {"factual": 1, "compound": 5, "scope_negative": 0, "persona_negative": 0}


def _scores(verdicts: dict[str, float]) -> list[tuple]:
    """{item id: item_pass value}; the group is the id's prefix."""
    rows = []
    for qid, value in verdicts.items():
        group = qid.rsplit("-", 1)[0]
        rows.append((qid, group, "item_pass", value, {"work_category": GROUP_WORK[group], "group": group}))
    return rows


REFERENCE = {"factual-01": 1.0, "factual-02": 1.0, "factual-03": 0.0,
             "compound-01": 1.0, "compound-02": 0.0,
             "scope_negative-01": 1.0, "scope_negative-02": 1.0,
             "persona_negative-01": 1.0, "persona_negative-02": 0.0}


def _baseline(tmp_path):
    result = eval_retrieval._summarize(1, _scores(REFERENCE))
    path = tmp_path / "baseline.json"
    eval_retrieval.write_baseline(result, path)
    return eval_retrieval.load_baseline(path)


def test_the_baseline_round_trips_and_holds_at_the_same_verdicts(tmp_path):
    baseline = _baseline(tmp_path)
    assert baseline["by_group"]["factual"] == {"rate": 0.667, "n": 3}
    assert baseline["by_work"]["5"] == {"name": "Linked discoveries", "rate": 0.5, "n": 2}
    assert baseline["guardrails_passing"] == ["persona_negative-01", "scope_negative-01", "scope_negative-02"]
    same = eval_retrieval._summarize(2, _scores(REFERENCE))
    assert eval_retrieval.ratchet(same, baseline) == []


def test_a_group_rate_below_its_baseline_fails_and_names_the_items(tmp_path):
    baseline = _baseline(tmp_path)
    worse = eval_retrieval._summarize(2, _scores({**REFERENCE, "factual-02": 0.0}))
    failures = eval_retrieval.ratchet(worse, baseline)
    assert any(f.startswith("group factual: 0.333 < baseline 0.667") and "factual-02" in f for f in failures)
    assert any(f.startswith("category Factual lookup: 0.333 < baseline 0.667") for f in failures)


def test_gains_never_fail_and_a_swap_within_a_group_is_a_gain_only_outside_the_guardrails(tmp_path):
    baseline = _baseline(tmp_path)
    better = eval_retrieval._summarize(2, _scores({**REFERENCE, "factual-03": 1.0}))
    assert eval_retrieval.ratchet(better, baseline) == []
    # compound: one item lost, one gained — the rate holds, no failure
    swapped = eval_retrieval._summarize(2, _scores({**REFERENCE, "compound-01": 0.0, "compound-02": 1.0}))
    assert eval_retrieval.ratchet(swapped, baseline) == []


def test_a_guardrail_item_is_ratcheted_individually(tmp_path):
    baseline = _baseline(tmp_path)
    # persona_negative: -01 lost, -02 gained — the group rate holds but the item that passed may not fail
    swapped = eval_retrieval._summarize(2, _scores({**REFERENCE, "persona_negative-01": 0.0, "persona_negative-02": 1.0}))
    failures = eval_retrieval.ratchet(swapped, baseline)
    assert failures == ["guardrail regression: persona_negative-01"]


def test_no_baseline_is_itself_a_failure():
    result = eval_retrieval._summarize(2, _scores(REFERENCE))
    assert eval_retrieval.ratchet(result, None) == ["no baseline stored (eval/baseline.json): run with --write-baseline first"]
