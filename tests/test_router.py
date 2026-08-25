"""Router fixtures: query -> expected filters or boundary response."""

import pytest

from raglab.router import Route, route


IN_SCOPE_CASES = [
    # (query, expected_years, expected_plan_codes)
    ("What is the deductible?", (2026,), ()),
    ("What is the HDHP deductible?", (2026,), ("71-014",)),
    ("What was the High Option deductible in 2024?", (2024,), ("71-006",)),
    ("How did the deductible change from 2025 to 2026?", (2025, 2026), ()),
    ("How did High Option coinsurance change for 2026?", (2025, 2026), ("71-006",)),
    ("Did the HDHP out-of-pocket maximum change for 2026?", (2025, 2026), ("71-014",)),
    ("What was the deductible in 2021, and how does it compare to 2026?",
     (2021, 2026), ()),
    ("What is the Elevate Plus deductible?", (2026,), ("71-018",)),
    ("What is the postal HDHP deductible?", (2026,), ("71-026",)),
    ("How does GEHA coordinate benefits with Medicare?", (2026,), ()),
    # 71-006's product name, not a generic phrase
    ("How did the GEHA Benefit Plan deductible change from 2025 to 2026?",
     (2025, 2026), ("71-006",)),
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
        plan_codes=("71-014",),
        reasons=route(q).reasons,
    )
