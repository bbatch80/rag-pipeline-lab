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


def _unverified(verdicts: dict[str, float], skipped: set[str]) -> list[tuple]:
    """The same run with some items unverified (a check skipped: no warehouse)."""
    rows = []
    for qid, group, metric, value, detail in _scores(verdicts):
        rows.append((qid, group, "item_unverified" if qid in skipped else metric, 1.0 if qid in skipped else value, detail))
    return rows


def test_a_run_that_verifies_a_subset_is_compared_over_that_subset(tmp_path):
    """CI holds no warehouse credentials: the warehouse-backed items are
    unverified there. The baseline rate is recomputed over the items the
    run did verify, so a subset that matches the baseline item for item
    passes even when the whole-bucket rate looks lower."""
    baseline = _baseline(tmp_path)
    assert baseline["items"]["compound-01"] == 1 and baseline["items"]["compound-02"] == 0
    # compound at the baseline: 1/2. This run verifies only compound-02 (which failed at the baseline too): 0/1
    subset = eval_retrieval._summarize(2, _unverified(REFERENCE, {"compound-01"}))
    assert subset.by_group["compound"]["n"] == 1 and subset.by_group["compound"]["unverified"] == 1
    assert eval_retrieval.ratchet(subset, baseline) == []
    # ... but the verified item regressing still fails
    worse = eval_retrieval._summarize(2, _unverified({**REFERENCE, "factual-01": 0.0}, {"compound-01"}))
    failures = eval_retrieval.ratchet(worse, baseline)
    assert any(f.startswith("group factual: 0.333 < baseline 0.667 over the 3 items verified") for f in failures)


def test_an_unverified_guardrail_is_not_a_regression(tmp_path):
    baseline = _baseline(tmp_path)
    skipped = eval_retrieval._summarize(2, _unverified(REFERENCE, {"scope_negative-01"}))
    assert "scope_negative-01" not in skipped.guardrails_passing
    assert eval_retrieval.ratchet(skipped, baseline) == []


def test_a_new_item_unknown_to_the_baseline_is_ignored_by_the_ratchet(tmp_path):
    baseline = _baseline(tmp_path)
    grown = eval_retrieval._summarize(2, _scores({**REFERENCE, "factual-04": 0.0}))
    assert eval_retrieval.ratchet(grown, baseline) == []
