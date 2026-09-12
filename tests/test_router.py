"""Router fixtures: query -> expected filters or boundary response."""

import pytest

from raglab.router import Route, route


IN_SCOPE_CASES = [
    # (query, expected_years, expected_plan_codes)
    ("What is the deductible?", (2026,), ()),
    ("What is the HDHP deductible?", (2026,), ("71-014", "71-026")),  # option without a program: both programs' HDHP
    ("What was the High Option deductible in 2024?", (2024,), ("71-006",)),
    ("How did the deductible change from 2025 to 2026?", (2025, 2026), ()),
    ("How did High Option coinsurance change for 2026?", (2025, 2026), ("71-006", "71-021")),
    ("Did the HDHP out-of-pocket maximum change for 2026?", (2025, 2026), ("71-014", "71-026")),
    ("What was the deductible in 2021, and how does it compare to 2026?",
     (2021, 2026), ()),
    ("What is the Elevate Plus deductible?", (2026,), ("71-018",)),
    ("What is the postal HDHP deductible?", (2026,), ("71-026",)),
    ("How does GEHA coordinate benefits with Medicare?", (2026,), ()),
    # 71-006's product name, not a generic phrase
    ("How did the GEHA Benefit Plan deductible change from 2025 to 2026?",
     (2025, 2026), ("71-006", "71-021")),
    # "plan" phrasing routes like "option" phrasing
    ("For the 2025 FEHB standard plan, who is covered under Self and Family?",
     (2025,), ("71-006",)),
    ("What does the high plan pay for urgent care?", (2026,), ("71-006", "71-021")),
]


@pytest.mark.parametrize("query,years,plans", IN_SCOPE_CASES)
def test_in_scope_routing(query, years, plans):
    decision = route(query)
    assert decision.scope == "in_scope", decision
    assert decision.years == years
    assert decision.plan_codes == plans


BOUNDARY_CASES = [
    ("What is the deductible for Blue Cross FEP Standard?", "out_of_domain"),
    ("What is the standard monthly Medicare Part B premium for 2026?", "out_of_domain"),
    ("What was the GEHA High Option deductible in 2019?", "out_of_year"),
    ("What will the premium be in 2027?", "out_of_year"),
]


@pytest.mark.parametrize("query,expected_scope", BOUNDARY_CASES)
def test_boundary_routing(query, expected_scope):
    decision = route(query)
    assert decision.scope == expected_scope, decision
    assert decision.boundary_response, "boundary verdicts must carry a response"
    assert decision.years == () and decision.plan_codes == ()


def test_route_is_deterministic():
    q = "How did the HDHP deductible change for 2026?"
    assert route(q) == route(q) == Route(
        scope="in_scope",
        years=(2025, 2026),
        plan_codes=("71-014", "71-026"),
        reasons=route(q).reasons,
        change=True,
        cover_field="plan_code", cover_keys=("71-014", "71-026"), cover_asked="HDHP", cover_level="option",
    )


def test_route_is_enforce_over_read():
    """The router is two halves: a reading of the question (regex here, the
    planner model in the pipeline) and deterministic enforcement."""
    from raglab.router import Reading, enforce, read
    for query, *_ in IN_SCOPE_CASES if "IN_SCOPE_CASES" in globals() else []:
        assert route(query) == enforce(read(query))
    reading = read("For FEHB High, does the member have to pick a PCP?")
    assert reading.program == "FEHB" and reading.options == (), "the regex reader misses bare 'High' (factual-05)"
    model = Reading(program="FEHB", options=("high",), origin="model")
    r = enforce(model)
    assert r.plan_codes == ("71-006",) and r.cover_field is None and "reading: model" in r.reasons
    r = enforce(Reading(scope="other_carrier", boundary_value="aetna", origin="model"))
    assert r.scope == "out_of_domain" and "aetna" in r.boundary_response
    r = enforce(Reading(scope="medicare_program", origin="model"))
    assert r.scope == "out_of_domain" and "Medicare" in r.boundary_response
    r = enforce(Reading(years=(2019,), origin="model"))
    assert r.scope == "out_of_year" and "2019" in r.boundary_response
    r = enforce(Reading(options=("hdhp",), origin="model"))
    assert set(r.cover_keys) == {"71-014", "71-026"}, "option without a program still covers both programs"


def test_a_boundary_without_a_name_is_not_a_boundary():
    """The reader said 'other carrier' about a claims bulletin because it
    saw the word 'examiners' (run 506, 5 false refusals). No name, no boundary."""
    from raglab.router import Reading, enforce
    r = enforce(Reading(scope="other_carrier", boundary_value=None, years=(2026,), origin="model"))
    assert r.scope == "in_scope" and r.years == (2026,)
    assert any("named no carrier" in reason for reason in r.reasons)
    r = enforce(Reading(scope="other_carrier", boundary_value="Aetna", origin="model"))
    assert r.scope == "out_of_domain" and "Aetna" in r.boundary_response
