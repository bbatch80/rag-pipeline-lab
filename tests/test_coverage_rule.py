"""The coverage rule (Phase 3.5): the registry declares each source's metadata
hierarchy; when a question names one level but not the level beneath, the
platform searches every key beneath, keeps the best evidence per key, and
states coverage in the payload. One rule — no question-type cases."""
import json

import jsonschema
import pytest

from raglab import config, payload, rerank, retrieval, router
from raglab.retrieval import Candidate

SCHEMA = json.loads((config.REPO_ROOT / "db" / "payload.schema.json").read_text())


def test_program_named_without_a_plan_covers_every_plan_of_the_program():
    r = router.route("How many chiropractic services are allowed on the PSHB plan?")
    assert r.cover_field == "plan_code" and r.cover_asked == "PSHB"
    assert set(r.cover_keys) == {"71-021", "71-022", "71-026"} and set(r.plan_codes) == set(r.cover_keys)
    assert any("cover every PSHB plan" in reason for reason in r.reasons)


def test_a_named_plan_or_no_program_does_not_trigger_coverage():
    assert router.route("What does the PSHB High Option cover for chiropractic care?").cover_field is None
    assert router.route("How many chiropractic visits does the plan allow?").cover_field is None  # no program named
    r = router.route("What is the FEHB deductible?")
    assert r.cover_field == "plan_code" and r.cover_asked == "FEHB" and set(r.cover_keys) == {"71-006", "71-014", "71-018"}


def test_option_named_without_a_program_covers_every_plan_offering_it():
    """factual-03 (2026-09-11): 'HDHP' exists under FEHB and PSHB; naming the
    option but not the program must not guess FEHB."""
    r = router.route("Who handles the HSA for HDHP?")
    assert r.cover_field == "plan_code" and r.cover_level == "option" and r.cover_asked == "HDHP"
    assert set(r.cover_keys) == {"71-014", "71-026"} and set(r.plan_codes) == set(r.cover_keys)
    r = router.route("What is the High Option specialist copay?")
    assert set(r.cover_keys) == {"71-006", "71-021"} and r.cover_level == "option"
    # Elevate: the PSHB contract ended after 2025, so an undated (2026) question has one plan and no coverage.
    r = router.route("Does Elevate Plus cover out-of-network care?")
    assert r.cover_field is None and r.plan_codes == ("71-018",)
    r = router.route("Did Elevate Plus cover out-of-network care in 2025?")
    assert set(r.cover_keys) == {"71-018", "71-022"}
    # A program cue settles it: no coverage across programs.
    assert router.route("What is the FEHB HDHP deductible?").cover_field is None
    assert router.route("What is the PSHB HDHP deductible?").plan_codes == ("71-026",)


def test_option_coverage_reads_the_hierarchy_and_yields_to_enrollment():
    r = router.route("Who handles the HSA for HDHP?", hierarchies={"brochure": ("plan_code", "year")})
    assert r.cover_field is None and r.plan_codes == ("71-014",), "no program level declared -> nothing above to cover"
    covered = router.route("Who handles the HSA for HDHP?")
    ctx = retrieval.Context(member_key="k", enrollment={2026: "71-026"})
    bound = retrieval.bind_enrollment_plan(covered, ctx)
    assert bound.plan_codes == ("71-026",) and bound.cover_field is None and bound.plan_from_enrollment
    other = retrieval.bind_enrollment_plan(covered, retrieval.Context(member_key="k", enrollment={2026: "71-006"}))
    assert other.cover_keys == covered.cover_keys, "an enrollment outside the covered plans leaves the question's coverage alone"


def test_the_rule_reads_the_declared_hierarchy_not_a_question_type():
    """Remove 'plan_code' from the brochure hierarchy and the rule has nothing
    to cover — the behavior comes from the registry's declaration."""
    r = router.route("How many chiropractic services are allowed on the PSHB plan?", hierarchies={"brochure": ("program", "year")})
    assert r.cover_field is None and r.plan_codes == ()


def _cand(i, plan, score):
    return Candidate(chunk_id=i, content=f"c{i}", section="", doc_title=f"doc {plan}", source_path="p", plan_code=plan,
                     year=2026, acl_tag="public", pages=[1], vector_rank=1, text_rank=1, rrf_score=0.03, rerank_score=score,
                     content_hash="h", doc_type="brochure")


def test_rerank_interleaves_covered_plans_so_each_plan_leads(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda model, pairs: [c for c in [0.9, 0.85, 0.8, 0.4, 0.3, 0.2]])
    monkeypatch.setattr(rerank, "_get_model", lambda: object())
    pool = [_cand(1, "71-021", 0), _cand(2, "71-021", 0), _cand(3, "71-021", 0), _cand(4, "71-026", 0), _cand(5, "71-026", 0), _cand(6, "71-022", 0)]
    out = rerank.rerank("q", pool, top_n=4, stratify_plans=("71-021", "71-022", "71-026"))
    assert {c.plan_code for c in out[:3]} == {"71-021", "71-022", "71-026"}, "every covered plan's best chunk is in the top three"
    assert out[0].plan_code == "71-021", "the strongest plan leads; the others still get their seats"
    plain = rerank.rerank("q", pool, top_n=4)
    assert [c.plan_code for c in plain[:3]] == ["71-021", "71-021", "71-021"], "without the rule the dominant plan fills the top"


def test_each_lane_gets_one_seat_then_score_order(monkeypatch):
    """Two covered plans plus a weak no-plan chunk: every lane's best is
    seated, and the remaining seats go by score — a 0.47 letter is not
    placed above 0.85 brochure chunks, and the second brochure does not
    take half the list (F1/T4/Y1/Y8 regression, 2026-09-11)."""
    monkeypatch.setattr(rerank, "score_pairs", lambda model, pairs: [0.99, 0.98, 0.85, 0.70, 0.60, 0.47])
    monkeypatch.setattr(rerank, "_get_model", lambda: object())
    letter = _cand(6, None, 0); letter.doc_type = "carrier_letter"
    pool = [_cand(1, "71-014", 0), _cand(2, "71-026", 0), _cand(3, "71-014", 0), _cand(4, "71-014", 0), _cand(5, "71-026", 0), letter]
    out = rerank.rerank("q", pool, top_n=5, stratify_plans=("71-014", "71-026"))
    assert [c.chunk_id for c in out] == [1, 2, 3, 4, 5], "seats: 71-014 best, 71-026 best; then score order — the 0.47 letter owns no seat"


def test_interleave_never_demotes_chunks_from_sources_without_a_plan(monkeypatch):
    """P4 regression: a claims bulletin (no plan code) scoring highest must
    lead the list even when the rule covers three plans."""
    monkeypatch.setattr(rerank, "score_pairs", lambda model, pairs: [0.95, 0.8, 0.7, 0.6, 0.5])
    monkeypatch.setattr(rerank, "_get_model", lambda: object())
    bulletin = _cand(9, None, 0); bulletin.doc_type = "bulletin"
    pool = [bulletin, _cand(1, "71-021", 0), _cand(2, "71-021", 0), _cand(3, "71-026", 0), _cand(4, "71-022", 0)]
    out = rerank.rerank("q", pool, top_n=4, stratify_plans=("71-021", "71-022", "71-026"))
    assert out[0].chunk_id == 9, "the strongest chunk leads even with no plan"
    assert {c.plan_code for c in out} >= {"71-021", "71-022", "71-026"}, "every covered plan still has its seat"


def test_payload_coverage_note_validates_and_names_the_gap():
    r = router.route("How many chiropractic services are allowed on the PSHB plan?")
    note = {"field": "plan_code", "asked": "PSHB", "keys": list(r.cover_keys), "in_corpus": ["71-021", "71-026"],
            "with_evidence": ["71-021", "71-026"], "missing_from_corpus": ["71-022"]}
    built = payload.build("q", r, [_cand(1, "71-021", 0.9), _cand(2, "71-026", 0.8)], coverage=note)
    jsonschema.validate(built, SCHEMA)
    assert built["coverage"]["missing_from_corpus"] == ["71-022"]
    assert "coverage" not in payload.build("q", router.route("What is the High Option deductible?"), [_cand(1, "71-006", 0.9)])


def test_search_runs_once_per_covered_plan(db, monkeypatch):
    """Under the fixture: three brochure documents with different plan codes;
    a covered search returns candidates from each plan, a plain search may not."""
    from test_governance import _seed_tiers
    _seed_tiers(db, per_tier=0, embed=True)
    vec = "[" + ",".join(["0.5"] * 1536) + "]"
    for i, code in enumerate(("71-021", "71-022", "71-026")):
        doc = db.execute("INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) VALUES (%s, %s, 'h', 'public', 1) RETURNING id",
                         (f"t/{code}.pdf", f"GEHA PSHB {code} 2026")).fetchone()[0]
        for j in range(3 if code == "71-021" else 1):
            db.execute("INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, plan_code, doc_type, embedding) "
                       "VALUES (%s, %s, %s, 'public', 2026, %s, 'brochure', %s)", (doc, j, f"chiropractic services {code} {j}", code, vec))
    db.execute("SET LOCAL enable_indexscan = off")
    route = router.route("How many chiropractic services are allowed on the PSHB plan?")
    out = retrieval.search(db, "chiropractic services", vec, route, embed=lambda t: vec)
    assert {c.plan_code for c in out} == {"71-021", "71-022", "71-026"}


@pytest.mark.readonly
def test_every_reading_gets_a_full_pool_and_the_cap_is_stated(db, monkeypatch):
    """Ambiguity costs time, never recall (2026-09-12): two covered plans ×
    two routed years = four readings, each searched with the FULL budget
    (the old split gave each plan-year 15 of 50); the merged pool is cut at
    the cap and the payload's search note says so."""
    calls = []

    def fake_hybrid(conn, lexical, vector, where, params, fused_limit, weight=1.0, sources=()):
        calls.append(fused_limit)
        base = len(calls) * 1000
        return [_cand(base + i, "71-006", 0) for i in range(fused_limit)]

    monkeypatch.setattr(retrieval, "_hybrid", fake_hybrid)
    monkeypatch.setattr(retrieval, "_ingested_sources", lambda conn: ("brochure",))
    monkeypatch.setattr(retrieval, "_floor_sources", lambda conn, route: [])
    monkeypatch.setattr(retrieval, "MAX_CANDIDATES", 120)
    route = router.route("How did the High Option deductible change from 2025 to 2026?")
    assert set(route.cover_keys) == {"71-006", "71-021"} and route.years == (2025, 2026)
    stats = {}
    pool = retrieval.search(db, "q", "[0]", route, fused_limit=50, embed=lambda t: "[0]", stats=stats)
    assert calls == [50, 50, 50, 50], "four readings, each with the full budget"
    assert stats["readings"] == 4 and stats["cap"] == 120
    assert len(pool) == 120 and stats["candidates"] == 120 and stats["trimmed"] == 80
    assert len({c.chunk_id for c in pool[:4]}) == 4 and pool[0].chunk_id // 1000 != pool[1].chunk_id // 1000, "round-robin: each reading keeps its top share"


def test_payload_carries_the_search_note():
    built = payload.build("q", router.route("What is the HDHP deductible?"), [],
                          search={"readings": 2, "candidates": 100, "trimmed": 0, "cap": 600})
    assert built["search"]["readings"] == 2
    jsonschema.validate(built, SCHEMA)
