"""The row-identity rule: a lookup table answers by having the row. A chunk
is a table when it carries a Markdown separator row; its keys are the data
rows' first cells; a key the question names exactly seats the chunk by
identity with the score left honest. Prose, headers, numbers, and plan
words never match."""

from raglab import payload as payload_mod
from raglab import planner, router
from raglab.retrieval import Candidate

FORMULARY = ("This chunk is from Formulary 2026, section 'Formulary 2026 — Plan Year Reference'.\n"
             "| Drug | Class | Tier | Prior Auth |\n|---|---|---|---|\n"
             "| atorvastatin | statin | 1 | No |\n| metformin | biguanide | 1 | No |\n"
             "| Jardiance (empagliflozin) | SGLT2 inhibitor | 2 | Yes |\n| insulin glargine (Basaglar) | long-acting insulin | 1 | No |\n")
BROCHURE_TABLE = "| Benefit | High Option | Standard Option |\n|---|---|---|\n| Specialist visits | $45 | $50 |\n| High Option | x | y |\n| 2026 | a | b |\n| Urgent care | $50 | $60 |\n"


def _cand(cid, content, score, doc_type="formulary"):
    return Candidate(chunk_id=cid, content=content, section="", doc_title=f"{doc_type}-{cid}", source_path="", plan_code=None, year=2026,
                     acl_tag="employee", pages=[1], vector_rank=1, text_rank=1, rrf_score=0.03, rerank_score=score, doc_type=doc_type)


def test_table_keys_are_the_data_rows_first_cells():
    assert planner.table_keys(FORMULARY) == ["atorvastatin", "metformin", "jardiance empagliflozin", "insulin glargine basaglar"]
    assert planner.table_keys(BROCHURE_TABLE) == ["specialist visits", "urgent care"]  # header, a plan word, and a number are not keys
    assert planner.table_keys("Tier 1 generic drugs such as metformin | are covered.") == []   # a bar in prose is not a table
    prose_table = ("| Benefit | High Option |\n|---|---|\n| Notes: | see below |\n"
                   "| Removal of tumors and cysts, correction of congenital anomalies limited to children under 18 unless there is a functional deficit | $45 |\n"
                   "| Professional services of physicians in the office, medical consultations, second surgical opinions | $45 |\n| Treatment of burns | $45 |\n")
    assert planner.table_keys(prose_table) == []        # unanswerable-05: sentences laid out as a table are not lookup rows; 'Notes:' is a label
    assert planner.table_keys("| Drug | Tier |\n|---|---|\n| metformin | 1 |\n") == []   # one row is not a lookup table
    assert planner.table_keys("") == []


def test_the_question_must_name_the_row_exactly():
    assert planner.row_matches("Which drug tier is metformin?", FORMULARY) == ["metformin"]
    assert planner.row_matches("Is insulin glargine (Basaglar) covered?", FORMULARY) == ["insulin glargine basaglar"]
    assert planner.row_matches("Which drug tier is ibuprofen?", FORMULARY) == []                # unlisted: no row, no identity
    assert planner.row_matches("What is the specialist copay?", BROCHURE_TABLE) == []          # 'specialist' is not the cell 'specialist visits'
    assert planner.row_matches("What do I pay for specialist visits?", BROCHURE_TABLE) == ["specialist visits"]
    assert planner.row_matches("What is the High Option deductible?", BROCHURE_TABLE) == []    # plan words alone never match
    assert "drug" not in planner.question_phrases("Which drug tier is metformin?") and "metformin" in planner.question_phrases("Which drug tier is metformin?")


def test_a_matching_row_seats_the_chunk_by_identity_and_the_score_stays_honest():
    core = _cand(1, FORMULARY, 0.456)
    other = _cand(2, "| Drug | Tier |\n|---|---|\n| glipizide | 1 |\n", 0.6)
    prose = _cand(3, "Prescription drug tiers: generic, preferred, non-preferred.", 0.7, doc_type="brochure")
    reranked = [prose, other]                      # the reranker seated prose and the wrong table; the metformin chunk fell out
    seated, n, keys = planner.apply_row_identity(reranked, [core, other, prose], "Which drug tier is metformin?", top_n=8)
    assert [c.chunk_id for c in seated] == [1, 3, 2] and n == 1 and keys == ("metformin",)
    assert planner.apply_row_identity(reranked, [core, other, prose], "Which drug tier is ibuprofen?", top_n=8) == (reranked, 0, ())
    route = router.Route(scope="in_scope", years=(2026,))
    built = payload_mod.build("Which drug tier is metformin?", route, [core], identity_evidence=1)
    assert built["status"] == "ok" and built["confidence"] == 0.456 and built["evidence_by_identity"] == 1
    assert payload_mod.build("q", route, [core])["status"] == "insufficient_evidence"
