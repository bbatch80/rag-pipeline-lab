"""Retrieval contract tests: RRF math, rare-lexeme query building,
relevance matching, rerank/abstention mechanics (fake model — no weights)."""

import pytest


from raglab import ablation, rerank
from raglab.retrieval import Candidate, RRF_K, search
from raglab.router import Route

pytestmark = pytest.mark.readonly


def _candidate(**kw):
    base = dict(
        chunk_id=1, content="x", section="", doc_title="t", source_path="p",
        plan_code=None, year=2026, acl_tag="public", pages=[],
        vector_rank=None, text_rank=None, rrf_score=0.0,
    )
    base.update(kw)
    return Candidate(**base)


@pytest.fixture
def tiny_corpus(db):
    doc_id = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id) "
        "VALUES ('t/rrf.pdf', 'RRF Fixture', 'h', 1) RETURNING id"
    ).fetchone()[0]
    # Three chunks with hand-built vectors: chunk A nearest to the query
    # vector, chunk C matches the rare term 'zephyrite'. Vectors vary in
    # DIRECTION (constant vectors are all parallel — cosine distance 0).
    def vec(x):
        rest = ",".join(["0.0"] * 1534)
        return f"[1.0,{x:.3f},{rest}]"

    contents = [
        ("alpha benefit text", vec(0.100)),          # A: vector rank 1
        ("beta benefit text", vec(0.300)),           # B: vector rank 2
        ("gamma zephyrite identifier", vec(0.900)),  # C: vector rank 3
    ]
    for i, (content, v) in enumerate(contents):
        db.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, year, doc_type, embedding) "
            "VALUES (%s, %s, %s, 2026, 'brochure', %s::vector)",
            (doc_id, i, content, v),
        )
    return db, vec


@pytest.mark.clean_corpus
def test_rrf_math_matches_hand_computation(tiny_corpus):
    db, vec = tiny_corpus
    route = Route(scope="in_scope", years=(2026,))
    results = search(db, "zephyrite", vec(0.100), route)
    by_id = {c.content.split()[0]: c for c in results}

    # A: vector rank 1, no text match -> 1/(60+1)
    assert by_id["alpha"].rrf_score == pytest.approx(1 / (RRF_K + 1))
    # C: text rank 1 (only 'zephyrite' match) + vector rank 3
    assert by_id["gamma"].text_rank == 1
    assert by_id["gamma"].rrf_score == pytest.approx(
        1 / (RRF_K + 3) + 1 / (RRF_K + 1)
    )
    # Fusion promotes C (both arms) above A (one arm, better rank).
    assert results[0].content.startswith("gamma")




def test_is_relevant_brochure_page_offset():
    c = _candidate(plan_code="71-006", year=2026, pages=[16])
    assert ablation.is_relevant(c, [{"plan_code": "71-006", "year": 2026,
                                     "printed_pages": [14]}])
    assert not ablation.is_relevant(c, [{"plan_code": "71-006", "year": 2025,
                                         "printed_pages": [14]}])
    assert not ablation.is_relevant(c, [{"plan_code": "71-014", "year": 2026,
                                         "printed_pages": [14]}])


def test_is_relevant_internal_suffix():
    c = _candidate(source_path="data/internal/kb/kb_mail_order.md")
    assert ablation.is_relevant(c, [{"internal": "kb/kb_mail_order.md"}])
    assert not ablation.is_relevant(c, [{"internal": "kb/kb_other.md"}])


class _FakeModel:
    def predict(self, pairs):
        # Score by presence of 'answer' in the chunk text.
        return [0.9 if "answer" in chunk else 0.1 for _, chunk in pairs]


def test_rerank_year_stratification(monkeypatch):
    class _YearBiasModel:
        def predict(self, pairs):
            # 2026 chunks always outscore 2025 ones — the crowding-out setup.
            return [0.9 if "y2026" in chunk else 0.6 for _, chunk in pairs]

    monkeypatch.setattr(rerank, "_model", _YearBiasModel())
    cands = [
        _candidate(chunk_id=i, year=2026, content=f"y2026 chunk {i}")
        for i in range(6)
    ] + [
        _candidate(chunk_id=10 + i, year=2025, content=f"y2025 chunk {i}")
        for i in range(6)
    ]
    plain = rerank.rerank("q", list(cands), top_n=6)
    assert all(c.year == 2026 for c in plain), "unstratified: 2026 crowds out 2025"

    strat = rerank.rerank("q", list(cands), top_n=6, stratify_years=(2025, 2026))
    years = [c.year for c in strat]
    assert years.count(2025) == 3 and years.count(2026) == 3, (
        "stratified: both routed years hold their share of slots"
    )


def test_rerank_orders_and_verdicts(monkeypatch):
    monkeypatch.setattr(rerank, "_model", _FakeModel())
    cands = [
        _candidate(chunk_id=1, content="noise text"),
        _candidate(chunk_id=2, content="the answer text"),
    ]
    ordered = rerank.rerank("q", cands)
    assert ordered[0].chunk_id == 2
    abstain, best = rerank.abstention_verdict(ordered)
    assert not abstain and best == pytest.approx(0.9)

    ordered_noise = rerank.rerank("q", [_candidate(content="noise")])
    abstain, best = rerank.abstention_verdict(ordered_noise)
    assert abstain, "best score below threshold must abstain"


def test_filters_member_context_and_event_exemption():
    """Member-scoped sources are filtered to the member context or skipped
    without one; event sources pass the year (edition) filter."""
    from raglab.retrieval import _filters
    from raglab.router import Route

    route = Route(scope="in_scope", years=(2025,), plan_codes=(), sources=())
    where, params = _filters(route, None, ("call_note",), ("call_note",))
    assert "c.doc_type <> ALL(%s)" in where and ["call_note"] in params
    assert "(c.year = ANY(%s) OR c.doc_type = ANY(%s))" in where

    where, params = _filters(route, "person-key", ("call_note",), ("call_note",))
    assert "(c.doc_type <> ALL(%s) OR c.member_key = %s)" in where
    assert "person-key" in params


def _has_roster(db) -> bool:
    return db.execute("SELECT to_regclass('synthea.patients')").fetchone()[0] is not None


def test_resolve_member_without_roster_is_no_context(db):
    """A database without the synthea schema (CI) resolves nothing and
    leaves the transaction usable."""
    from raglab import retrieval

    if _has_roster(db):
        pytest.skip("roster present; the CI-shaped case needs a database without synthea")
    assert retrieval.resolve_member(db, "M822099594", "") is None
    assert db.execute("SELECT 1").fetchone()[0] == 1


def test_resolve_member_prefers_structured_context(db):
    """The structured member ID wins; an ID typed in the question is the
    fallback; names never resolve; an invalid ID is rejected."""
    from raglab import retrieval

    if not _has_roster(db):
        pytest.skip("needs the synthea roster (not in CI's fresh database)")
    row = db.execute("SELECT member_id, mrn, id FROM synthea.patients LIMIT 1").fetchone()
    other = db.execute("SELECT member_id FROM synthea.patients OFFSET 1 LIMIT 1").fetchone()[0]
    key = str(row[2])
    assert retrieval.resolve_member(db, row[0], f"what about member {other}?") == key
    assert retrieval.resolve_member(db, None, f"what did member {row[0]} call about?") == key
    assert retrieval.resolve_member(db, None, f"member with {row[1]} called") == key
    assert retrieval.resolve_member(db, None, "what did Karima Dickinson call about?") is None
    with pytest.raises(ValueError):
        retrieval.resolve_member(db, "M123", "")


def test_embed_cache_answers_repeat_questions(db, monkeypatch):
    """The second embedding of the same text is served from the table; a
    different text calls the embedder again."""
    from raglab import retrieval
    from raglab.eval_retrieval import EVAL_SCHEMA_PATH

    db.execute(EVAL_SCHEMA_PATH.read_text())
    calls = []
    monkeypatch.setattr(retrieval, "embed_query", lambda t: calls.append(t) or f"[vec:{t}]")
    monkeypatch.setattr(retrieval, "EMBED_CACHE", True)
    assert retrieval.embed_cached(db, "q one") == "[vec:q one]"
    assert retrieval.embed_cached(db, "q one") == "[vec:q one]"
    assert retrieval.embed_cached(db, "q two") == "[vec:q two]"
    assert calls == ["q one", "q two"]


def test_claim_id_resolves_to_its_member(db):
    from raglab import retrieval

    if not _has_roster(db):
        pytest.skip("needs the synthea call log (not in CI's fresh database)")
    call_id, patient, claim = db.execute(
        "SELECT call_id, patient, claim_id FROM synthea.call_log WHERE claim_id IS NOT NULL LIMIT 1"
    ).fetchone()
    assert retrieval.resolve_member(db, None, f"What happened with claim {claim}?") == str(patient)


def test_case_id_resolves_to_its_member(db):
    from raglab import retrieval

    if not _has_roster(db):
        pytest.skip("needs the synthea appeals table (not in CI's fresh database)")
    row = db.execute("SELECT case_id, patient FROM synthea.appeals LIMIT 1").fetchone()
    if row is None:
        pytest.skip("no appeals generated")
    assert retrieval.resolve_member(db, None, f"What was the basis of appeal {row[0]}?") == str(row[1])


def test_context_turns_identifiers_into_filters(db):
    """Identifiers are context: resolved to the member and the record, and
    removed from the ranking query."""
    from raglab import retrieval

    if not _has_roster(db):
        pytest.skip("needs the synthea roster (not in CI's fresh database)")
    case, patient, claim, policy = db.execute("SELECT case_id, patient, claim_id, policy_id FROM synthea.appeals LIMIT 1").fetchone()
    question = f"Why was appeal {case} overturned, and which policy applied?"
    ctx = retrieval.resolve_context(db, None, question)
    # A case names its claim and the policy applied (Phase 3 decision 5): all
    # three are record context, bound by key before any search or plan.
    assert ctx.member_key == str(patient)
    assert ctx.record["case_id"] == case and ctx.record["claim_id"] == claim
    assert ctx.record.get("policy_id") == policy if policy else "policy_id" not in ctx.record
    assert ctx.query.startswith(question), "the ranking query keeps the identifier (measured: stripping loses the ranker's cue)"
    assert ctx.as_of is not None, "the claim's date of service is bound from the case"
    ctx = retrieval.resolve_context(db, None, f"What happened with claim {claim}?")
    assert ctx.record["claim_id"] == claim and "case_id" not in ctx.record


def test_context_can_strip_identifiers_for_the_ab(db, monkeypatch):
    from raglab import retrieval

    if not _has_roster(db):
        pytest.skip("needs the synthea roster (not in CI's fresh database)")
    case = db.execute("SELECT case_id FROM synthea.appeals LIMIT 1").fetchone()[0]
    monkeypatch.setattr(retrieval, "STRIP_IDS", True)
    ctx = retrieval.resolve_context(db, None, f"Why was appeal {case} overturned, and which policy applied?")
    assert case not in ctx.query and "overturned" in ctx.query and ctx.query.endswith("?")
    assert retrieval.resolve_context(db, None, case).query == case  # an identifier alone keeps its text


def test_filters_record_context_narrows_member_scoped_sources_only():
    from raglab.retrieval import _filters
    from raglab.router import Route

    where, params = _filters(Route(scope="in_scope"), "k", ("appeal",), (), {"case_id": "APL-0000000"})
    assert "c.metadata->'record'->>%s = %s" in where and "case_id" in params and "APL-0000000" in params
    assert "(c.doc_type <> ALL(%s) OR" in where


def test_policy_id_is_record_context_without_a_member(db):
    from raglab import retrieval

    ctx = retrieval.resolve_context(db, None, "What are the criteria under CP-0003?")
    assert ctx.record == {"policy_id": "CP-0003"} and ctx.member_key is None
