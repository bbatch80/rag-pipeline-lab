"""Reranker bake-off v2: thresholds re-derived per model, ranking metrics,
the re-identified text axis keyed apart in the score cache."""
import pytest

from raglab import rerank, rerankbakeoff

pytestmark = pytest.mark.readonly


def _seed(db, run_label="t"):
    run_id = db.execute("INSERT INTO eval_runs (kind, config_label) VALUES ('retrieval', %s) RETURNING id", (run_label,)).fetchone()[0]
    rows = [
        ("A1", "factual", "wrong_abstention", 0, {"best_score": 0.91}),
        ("A2", "factual", "wrong_abstention", 0, {"best_score": 0.72}),
        ("C1", "call_note", "wrong_abstention", 0, {"best_score": 0.20}),
        ("U1", "unanswerable", "abstained", 1, {"best_score": 0.30}),
        ("U2", "unanswerable", "abstained", 1, {"best_score": 0.12}),
        ("A1", "factual", "hit@5", 1, {"top_scores": [0.91, 0.909, 0.5]}),
        ("A2", "factual", "hit@5", 1, {"top_scores": [0.72, 0.40, 0.1]}),
    ]
    import json
    for q, c, m, v, d in rows:
        db.execute("INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) VALUES (%s, %s, %s, %s, %s, %s)",
                   (run_id, q, c, m, v, json.dumps(d)))
    return run_id


def test_thresholds_derive_from_the_models_own_separation(db):
    run_id = _seed(db)
    t = rerankbakeoff.derive_thresholds(db, run_id)
    assert t["prose"] == 0.51 and t["separation"] == 0.42, "midpoint of lowest answered prose (0.72) and highest trap (0.30)"
    assert t["record"] == 0.16, "record bar = lowest answered record best (0.20) with a 20% margin"


def test_tie_rate_and_margin_read_the_top_scores(db):
    run_id = _seed(db)
    r = rerankbakeoff.ranking_metrics(db, run_id)
    assert r["n"] == 2 and r["tie_rate"] == 0.5, "A1's top two are within 0.005: a tie"
    assert abs(r["median_margin"] - statistics_median([0.001, 0.32])) < 1e-3


def statistics_median(xs):
    import statistics
    return statistics.median(xs)


def test_reidentified_scores_are_keyed_apart_in_the_cache(monkeypatch):
    monkeypatch.setattr(rerank, "_model_key", None)
    monkeypatch.setattr(rerank, "RERANK_REIDENTIFY", False)
    plain = rerank.model_key()
    monkeypatch.setattr(rerank, "RERANK_REIDENTIFY", True)
    assert rerank.model_key() == plain + "+reid"


def test_reidentify_replaces_vault_tokens_only_when_on(monkeypatch):
    monkeypatch.setattr(rerank, "_vault", {"[MEMBER_ID-0384]": "M123456789"})
    monkeypatch.setattr(rerank, "RERANK_REIDENTIFY", True)
    assert rerank._reidentify("member [MEMBER_ID-0384] called") == "member M123456789 called"
    assert rerank._reidentify("no tokens here") == "no tokens here"
    assert rerank._reidentify("[PERSON-9999] unknown") == "[PERSON-9999] unknown", "an unknown token stays a token"


def test_every_candidate_has_a_loader_kind_and_provisional_thresholds():
    for key in ("bge-base", "bge-v2-m3", "mxbai-large-v2", "qwen3-0.6b"):
        spec = rerank.RERANKERS[key]
        assert spec["kind"] in ("cross-encoder", "qwen3") and "threshold" in spec and "thresholds" in spec


def test_record_sources_share_the_record_bar():
    """Every member-scoped record source carries the record bar, not the
    prose bar (clinical notes were the exception until persona_negative-07)."""
    for doc_type in ("call_note", "appeal", "clinical_note"):
        assert rerank.threshold_for(doc_type) == 0.1, doc_type
    assert rerank.threshold_for("brochure") == 0.5
