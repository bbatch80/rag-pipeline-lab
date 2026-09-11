"""Two axes over the golden set: every item carries a group (its `category`,
which decides how it is scored) and a work category (which decides where the
dashboard reports it). The dashboard renders from whatever the store holds,
including nothing."""

import pytest

from raglab import ablation, dashboard, eval_retrieval, pipeline, router, taxonomy

pytestmark = pytest.mark.readonly


def test_every_golden_item_carries_both_axes():
    for item in ablation.load_golden():
        assert item["category"] in taxonomy.GROUPS, f"{item['id']}: unknown group {item['category']}"
        assert "work_category" in item, f"{item['id']}: no work category"
        assert int(item["work_category"]) in taxonomy.WORK_CATEGORIES, item["id"]
        assert taxonomy.work_category(item).band != "designed_out", (
            f"{item['id']}: a designed-out category cannot have gated items")


def test_descriptions_stay_general():
    """The category text must not name the corpus: it has to read the same
    after the corpus grows."""
    domain_words = ("GEHA", "brochure", "71-0", "Synthea", "Snowflake", "PSHB", "FEHB")
    for c in taxonomy.WORK_CATEGORIES.values():
        for word in domain_words:
            assert word not in c.name and word not in c.description, (c.number, word)


def test_item_verdicts_all_gate_metrics_must_be_one():
    scores = [
        ("A", "compound", "routing_accuracy", 1.0, {}),
        ("A", "compound", "complete_recall", 1.0, {}),
        ("B", "compound", "routing_accuracy", 1.0, {}),
        ("B", "compound", "complete_recall", 0.5, {}),
        ("C", "factual", "hit@5", 1.0, {}),
        ("C", "factual", "precision@5", 0.2, {}),  # not a gate metric: ignored
        ("D", "compound", "routing_accuracy", 1.0, {}),
        ("D", "compound", "complete_recall_skipped", 1.0, {}),  # skipped ≠ pass
    ]
    verdicts = taxonomy.item_verdicts(scores)
    assert verdicts["A"] == (True, [])
    assert verdicts["B"] == (False, [])
    assert verdicts["C"] == (True, [])
    assert verdicts["D"] == (None, ["complete_recall_skipped"])


def test_item_verdict_rows_carry_the_work_category():
    golden = {item["id"]: item for item in ablation.load_golden()}
    some_factual = next(q for q, i in golden.items() if i["category"] == "factual")
    some_compound = next(q for q, i in golden.items() if i["category"] == "compound")
    scores = [
        (some_factual, "factual", "hit@5", 1.0, {}),
        (some_compound, "compound", "routing_accuracy", 1.0, {}),
        (some_compound, "compound", "complete_recall_skipped", 1.0, {}),
        ("NOT-A-GOLDEN-ID", "factual", "hit@5", 1.0, {}),
    ]
    rows = {r[0]: r for r in eval_retrieval._item_verdict_rows(scores)}
    assert rows[some_factual][2] == "item_pass" and rows[some_factual][3] == 1.0
    assert rows[some_factual][4]["work_category"] == int(golden[some_factual]["work_category"])
    assert rows[some_compound][2] == "item_unverified"
    assert "NOT-A-GOLDEN-ID" not in rows


def test_summary_reports_pass_counts_per_work_category():
    golden = {item["id"]: item for item in ablation.load_golden()}
    factual = [q for q, i in golden.items() if i["category"] == "factual"][:2]
    scores = [(factual[0], "factual", "hit@5", 1.0, {}), (factual[1], "factual", "hit@5", 0.0, {})]
    scores += eval_retrieval._item_verdict_rows(scores)
    result = eval_retrieval._summarize(0, scores)
    number = int(golden[factual[0]]["work_category"])
    assert result.by_work[number]["n"] == 2 and result.by_work[number]["passed"] == 1


def test_dashboard_renders_both_axes(db, tmp_path):
    """Renders from whatever the store holds — including no verdict rows —
    with every category and group row present."""
    out = dashboard.render(db, tmp_path / "dashboard.html")
    html = out.read_text()
    for c in taxonomy.WORK_CATEGORIES.values():
        assert c.name in html
    for g in taxonomy.GROUPS.values():
        assert g.name in html
    assert "Open defects" in html and "Real questions" in html


def test_coverage_note_separates_not_offered_from_missing(db):
    """A plan the registry declares as ended is 'not offered', never
    'missing from corpus' (71-022 PSHB Elevate ended after 2025)."""
    decision = router.Route(scope="in_scope", years=(2026,), cover_field="plan_code",
                            cover_keys=("71-021", "71-022", "71-026"), cover_asked="PSHB")
    note = pipeline.coverage_note(db, decision, reranked=[])
    assert note["not_offered"] == ["71-022"]
    assert "71-022" not in note["missing_from_corpus"]
