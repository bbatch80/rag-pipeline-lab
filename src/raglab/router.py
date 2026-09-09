"""Rule-based metadata router: query analysis before any search touches the
database. Deterministic on purpose — every decision here is a testable fact
that the explain trace reports verbatim.

Policies:
- Scope gate: out-of-domain and out-of-year queries get a boundary verdict;
  retrieval is never attempted.
- Recency default: undated questions resolve to the current plan year.
- Explicit years route to those years; change/comparison language with a
  single year expands to (year-1, year).
- Plan filters pass NULL plan_code (internal docs span plans); year filters
  are strict.
"""

import os
import re
from dataclasses import dataclass, field

CURRENT_YEAR = 2026
CORPUS_YEARS = range(2021, 2027)

_OTHER_CARRIERS = (
    "blue cross", "bcbs", "fep", "aetna", "kaiser", "mhbp", "nalc",
    "apwu", "cigna", "humana", "anthem",
)
# Medicare coordination is in-corpus (Section 9); Medicare's own program
# facts (premiums, Part B amounts) are not.
_MEDICARE_OWN = re.compile(r"medicare.{0,40}(premium|part b cost|part d cost)|"
                           r"(premium|cost) for medicare", re.IGNORECASE)

_CHANGE_LANGUAGE = re.compile(
    r"\bchange[ds]?\b|\bcompared?\b|\bcompare[ds]?\b|\bdifference\b|\bvs\.?\b|"
    r"\bincrease[ds]?\b|\bdecrease[ds]?\b|\byear over year\b|\byoy\b",
    re.IGNORECASE,
)

_PLAN_PATTERNS = [
    (re.compile(r"\bhdhp\b|high.deductible", re.IGNORECASE), "hdhp"),
    (re.compile(r"\belevate plus\b", re.IGNORECASE), "elevate_plus"),
    (re.compile(r"\belevate\b", re.IGNORECASE), "elevate"),
    (re.compile(r"\bhigh (option|plan)\b", re.IGNORECASE), "high"),
    (re.compile(r"\bstandard (option|plan)\b", re.IGNORECASE), "standard"),
    # "GEHA Benefit Plan" is 71-006's product name (the brochure cover
    # title), not a generic phrase — it maps to the High/Standard brochure.
    (re.compile(r"\bbenefit plan\b", re.IGNORECASE), "high"),
]
_PSHB = re.compile(r"\bpshb\b|\bpostal\b", re.IGNORECASE)

_FEHB_CODES = {"hdhp": "71-014", "elevate": "71-018", "elevate_plus": "71-018",
               "high": "71-006", "standard": "71-006"}
_PSHB_CODES = {"hdhp": "71-026", "elevate": "71-022", "elevate_plus": "71-022",
               "high": "71-021", "standard": "71-021"}


@dataclass(frozen=True)
class Route:
    scope: str  # in_scope | out_of_domain | out_of_year
    boundary_response: str | None = None
    years: tuple[int, ...] = ()
    plan_codes: tuple[str, ...] = ()
    # doc_type keys to search; empty = every vector-lane source the caller
    # can see. Surfaces and cue families narrow it; RLS decides visibility.
    sources: tuple[str, ...] = ()
    reasons: tuple[str, ...] = field(default=())


def route(query: str) -> Route:
    reasons = []

    for carrier in _OTHER_CARRIERS:
        if carrier in query.lower():
            return Route(
                scope="out_of_domain",
                boundary_response=(
                    f"This corpus covers GEHA plans only; '{carrier}' is a "
                    "different carrier."
                ),
                reasons=(f"other-carrier term: {carrier!r}",),
            )
    if _MEDICARE_OWN.search(query):
        return Route(
            scope="out_of_domain",
            boundary_response=(
                "Medicare program facts (premiums, costs) are outside this "
                "corpus; it covers GEHA plan benefits, including how they "
                "coordinate with Medicare."
            ),
            reasons=("medicare-own-program pattern",),
        )

    mentioned_years = sorted(
        {int(y) for y in re.findall(r"\b(20\d{2})\b", query)}
    )
    out_of_range = [y for y in mentioned_years if y not in CORPUS_YEARS]
    if out_of_range:
        return Route(
            scope="out_of_year",
            boundary_response=(
                f"The corpus covers plan years {CORPUS_YEARS.start}-"
                f"{CORPUS_YEARS.stop - 1}; {out_of_range[0]} is outside it."
            ),
            reasons=(f"year out of range: {out_of_range[0]}",),
        )

    change = bool(_CHANGE_LANGUAGE.search(query))
    if mentioned_years:
        years = tuple(mentioned_years)
        reasons.append(f"explicit year(s): {years}")
        if change and len(years) == 1:
            years = (years[0] - 1, years[0])
            reasons.append("change language + single year -> include prior year")
            years = tuple(y for y in years if y in CORPUS_YEARS)
    elif change:
        years = (CURRENT_YEAR - 1, CURRENT_YEAR)
        reasons.append("change language, undated -> current + prior year")
    else:
        years = (CURRENT_YEAR,)
        reasons.append("undated -> recency default (current year)")

    program_codes = _PSHB_CODES if _PSHB.search(query) else _FEHB_CODES
    if _PSHB.search(query):
        reasons.append("PSHB program cue")
    plans = []
    for pattern, key in _PLAN_PATTERNS:
        if pattern.search(query) and program_codes[key] not in plans:
            # 'elevate plus' also matches the bare 'elevate' pattern; both
            # map to the same brochure, so dedupe handles it.
            plans.append(program_codes[key])
            reasons.append(f"plan cue: {key} -> {program_codes[key]}")

    return Route(
        scope="in_scope",
        years=years,
        plan_codes=tuple(plans),
        reasons=tuple(reasons),
    )


# Change language only — never function words ("in" would tear "in-network"
# apart at the hyphen). Dangling "from … to …" is handled with the years.
_CHANGE_WORDS = re.compile(
    r"\b(how did|how does|did|has|have|change[ds]?|changing|compared?(?: to| with)?|compare[ds]?|"
    r"difference(?: between)?|differ(?:s|ed)?|vs\.?|versus|increase[ds]?|decrease[ds]?|year over year|yoy)\b",
    re.IGNORECASE,
)
_YEAR_PHRASE = re.compile(r"\b(?:from|to|between|and|in|for|since|vs\.?|versus)\s+(?:19|20)\d{2}\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def year_neutral(query: str, year: int) -> str:
    """The per-year form of a change question: the subject without change
    language or year references, plus this year — "How did the High Option
    specialist copay change from 2025 to 2026?" → "High Option specialist
    copay 2025". A prior-year benefit table then competes on its subject
    instead of losing to the brochure's own "changes this year" section."""
    text = _YEAR_PHRASE.sub(" ", query)   # "from 2025", "to 2026", "in 2021"
    text = _YEAR.sub(" ", text)           # any bare year left
    text = _CHANGE_WORDS.sub(" ", text)
    text = re.sub(r"[?]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    return f"{text} {year}"


def year_queries(query: str, years: tuple[int, ...]) -> dict[int, str]:
    """Per-year queries for multi-year (change) routes; empty otherwise.
    The LATEST year keeps the original question: its answer is the
    brochure's own "changes this year" section, which the change wording
    finds. Prior years get the year-neutral form, so their benefit tables
    compete on the subject."""
    if len(years) < 2 or os.environ.get("RAGLAB_YEAR_NEUTRAL", "on") == "off":
        return {}
    latest = max(years)
    return {y: (query if y == latest else year_neutral(query, y)) for y in years}
