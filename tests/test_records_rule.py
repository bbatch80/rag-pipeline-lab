"""The records rule: when a member is open and the planner aimed a document
leg at a member-scoped source, the member's own records in that source are
seated first, in reranker order, and the leg never abstains on them; other
sources fill the remaining seats under their normal bar. Without a member,
without a member-scoped hint, or without any record in the pool, nothing
changes."""

from raglab import payload as payload_mod
from raglab import planner, router
from raglab.retrieval import Candidate


def _cand(cid, doc_type, score, title=None):
    return Candidate(chunk_id=cid, content=f"text {cid}", section="", doc_title=title or f"{doc_type}-{cid}", source_path="",
                     plan_code=None, year=2026, acl_tag="public", pages=[], vector_rank=1, text_rank=1, rrf_score=0.03,
                     rerank_score=score, doc_type=doc_type)


def test_member_records_are_seated_first_and_counted():
    note = _cand(1, "clinical_note", 0.004)
    policy = _cand(2, "clinical_policy", 0.015)
    brochure = _cand(3, "brochure", 0.012)
    reranked = [policy, brochure]              # what the reranker seated: the note fell out
    pool = [note, policy, brochure]
    seated, n = planner.apply_records_rule(reranked, pool, hinted={"clinical_note"}, member_key="k", top_n=8)
    assert [c.chunk_id for c in seated] == [1, 2, 3] and n == 1


def test_several_records_keep_reranker_order_and_fill_the_rest():
    a, b, c = _cand(1, "call_note", 0.2), _cand(2, "call_note", 0.9), _cand(3, "call_note", 0.5)
    other = [_cand(i, "brochure", 0.3) for i in range(10, 20)]
    seated, n = planner.apply_records_rule(other[:8], [a, b, c] + other, hinted={"call_note"}, member_key="k", top_n=8)
    assert [x.chunk_id for x in seated[:3]] == [2, 3, 1] and n == 3 and len(seated) == 8
    assert all(x.doc_type == "brochure" for x in seated[3:])


def test_the_rule_is_inert_without_a_member_a_hint_or_a_record():
    note = _cand(1, "clinical_note", 0.004)
    pool = [note, _cand(2, "brochure", 0.9)]
    ranked = [pool[1]]
    assert planner.apply_records_rule(ranked, pool, hinted={"clinical_note"}, member_key=None, top_n=8) == (ranked, 0)
    assert planner.apply_records_rule(ranked, pool, hinted=set(), member_key="k", top_n=8) == (ranked, 0)
    assert planner.apply_records_rule(ranked, [pool[1]], hinted={"clinical_note"}, member_key="k", top_n=8) == (ranked, 0)


def test_payload_status_is_ok_by_identity_and_says_so():
    route = router.Route(scope="in_scope", years=(2026,))
    note = _cand(1, "clinical_note", 0.004)
    without = payload_mod.build("q", route, [note])
    assert without["status"] == "insufficient_evidence" and "evidence_by_identity" not in without
    with_rule = payload_mod.build("q", route, [note], identity_evidence=1)
    assert with_rule["status"] == "ok" and with_rule["evidence_by_identity"] == 1
    assert with_rule["confidence"] == 0.004  # the score stays honest; the status is by identity


def test_a_fact_question_over_the_records_keeps_the_score_verdict():
    """unanswerable-05: 'Do their call notes say what their doctor prescribed?'
    The notes are seated (a rep sees them) but none carries the fact, so the
    verdict stays with the scores — no false ok."""
    notes = [_cand(i, "call_note", 0.01) for i in range(1, 6)]
    policy = _cand(9, "clinical_policy", 0.32)
    seated, n = planner.apply_records_rule([policy], notes + [policy], hinted={"call_note"}, member_key="k", top_n=8, open_ended=False)
    assert [c.chunk_id for c in seated[:5]] == [1, 2, 3, 4, 5] and n == 0
    assert planner._RECORDS_REQUEST.search("Can you tell me about this member's clinical history?")
    assert planner._RECORDS_REQUEST.search("Summarize every call this member made about appealing a denied claim.")
    assert planner._RECORDS_REQUEST.search("What calls are on file for her?")
    for q in ("What was the referral for?", "What did they call about with their new PCP?", "What was their latest clinical note about?",
              "What was done related to their ID card?", "What did the member say about the ER bill?"):
        assert planner._RECORDS_REQUEST.search(q) and not planner._NESTED_FACT.search(q), q  # open-ended: the record is the answer
    q = "Do their call notes say what their doctor prescribed?"
    assert planner._NESTED_FACT.search(q)  # a fact asked of the records: the verdict stays with the scores
