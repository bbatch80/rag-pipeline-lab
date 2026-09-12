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
_OPTION_LABEL = {"hdhp": "HDHP", "elevate": "Elevate", "elevate_plus": "Elevate Plus",
                 "high": "High Option", "standard": "Standard Option"}


def _plan_years() -> dict[str, set[int]]:
    """plan code -> the years the registry declares it offered (corpus.ALL_PLANS)."""
    from raglab import corpus
    return {spec.plan_code: set(spec.years) for spec in corpus.ALL_PLANS}


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
    # Versioned sources (clinical policies): the date whose in-effect version
    # applies. Explicit ("as of March 1, 2025", "in effect on 2026-01-06")
    # or None → the end of the routed year (undated → today's version).
    as_of: str | None = None
    change: bool = False  # change language: per-year / per-version search
    # The coverage rule (Phase 3.5): the question named a level of a source's
    # declared hierarchy but not the level beneath it — cover every key
    # beneath (one search per key, best evidence per key, coverage stated).
    # Today: program named, plan not -> cover_field 'plan_code', keys = that
    # program's plan codes. Years are covered by the multi-year mechanism.
    cover_field: str | None = None
    cover_keys: tuple[str, ...] = ()
    cover_asked: str | None = None  # the named level's value ('PSHB', 'HDHP')
    cover_level: str | None = None  # which level was named: 'program' (plans beneath covered) | 'option' (programs above covered)
    plan_from_enrollment: bool = False  # the plan filter came from the member's enrollment, not the question


# The registry declares these (sources.hierarchy, migration 024); the router
# falls back to this copy when called without a registry (pure callers, tests).
DEFAULT_HIERARCHIES = {"brochure": ("program", "plan_code", "year"), "clinical_policy": ("policy_id", "version"),
                       "carrier_letter": ("year",), "rates": ("year",)}
_FEHB = re.compile(r"\bfehb\b|\bfederal employees health benefits\b", re.IGNORECASE)


@dataclass(frozen=True)
class Reading:
    """What the question SAYS — the reading half of routing. Produced by the
    regex reader (`read`) or by the planner's model call (the model reads;
    the code enforces). Every value is a member of a registry-declared set."""
    program: str | None = None        # FEHB | PSHB | None
    options: tuple[str, ...] = ()     # plan option keys named: hdhp, elevate, elevate_plus, high, standard
    years: tuple[int, ...] = ()       # years explicitly mentioned
    change: bool = False              # change / comparison language
    as_of: str | None = None          # an explicit in-effect date, ISO
    scope: str = "in_scope"           # in_scope | other_carrier | medicare_program
    boundary_value: str | None = None  # the carrier named, when scope is other_carrier
    origin: str = "rules"             # rules | model


BOUNDARY_TEXT = {
    "other_carrier": "This corpus covers GEHA plans only; '{value}' is a different carrier.",
    "medicare_program": ("Medicare program facts (premiums, costs) are outside this corpus; it covers "
                         "GEHA plan benefits, including how they coordinate with Medicare."),
    "out_of_year": "The corpus covers plan years {first}-{last}; {value} is outside it.",
}


def read(query: str) -> Reading:
    """The regex reader: the fallback when the planner model did not read the
    question, and the eval's comparison."""
    lower = query.lower()
    for carrier in _OTHER_CARRIERS:
        if carrier in lower:
            return Reading(scope="other_carrier", boundary_value=carrier)
    if _MEDICARE_OWN.search(query):
        return Reading(scope="medicare_program")
    years = tuple(sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", query)}))
    options = tuple(dict.fromkeys(key for pattern, key in _PLAN_PATTERNS if pattern.search(query)))
    program = "PSHB" if _PSHB.search(query) else ("FEHB" if _FEHB.search(query) else None)
    return Reading(program=program, options=options, years=years, change=bool(_CHANGE_LANGUAGE.search(query)),
                   as_of=as_of_date(query), origin="rules")


def route(query: str, hierarchies: dict[str, tuple[str, ...]] | None = None,
          reading: Reading | None = None) -> Route:
    """The route for a question: enforce the data's declared shape on a
    reading of the question (the planner's, else the regex reader's)."""
    return enforce(reading or read(query), hierarchies)


def enforce(reading: Reading, hierarchies: dict[str, tuple[str, ...]] | None = None) -> Route:
    """The enforcing half: years default and change logic, plan codes,
    coverage across the declared hierarchy, boundary responses. Deterministic;
    never reads the question's words."""
    reasons = [f"reading: {reading.origin}"]
    if reading.scope == "other_carrier" and not (reading.boundary_value or "").strip():
        # A boundary must name what it is bounding. A reader that says "other
        # carrier" without naming one has guessed from a word ("carriers",
        # "examiners"): measured 2026-09-12, 5 of 101 readings — every one an
        # in-scope question. Enforcement: no name, no boundary.
        reasons.append("other-carrier reading named no carrier -> treated as in scope")
        reading = Reading(program=reading.program, options=reading.options, years=reading.years, change=reading.change,
                          as_of=reading.as_of, scope="in_scope", origin=reading.origin)
    if reading.scope == "other_carrier":
        return Route(scope="out_of_domain",
                     boundary_response=BOUNDARY_TEXT["other_carrier"].format(value=reading.boundary_value),
                     reasons=(f"reading: {reading.origin}", f"other-carrier term: {reading.boundary_value!r}"))
    if reading.scope == "medicare_program":
        return Route(scope="out_of_domain", boundary_response=BOUNDARY_TEXT["medicare_program"],
                     reasons=(f"reading: {reading.origin}", "medicare-own-program"))

    mentioned_years = sorted(reading.years)
    out_of_range = [y for y in mentioned_years if y not in CORPUS_YEARS]
    if out_of_range:
        return Route(
            scope="out_of_year",
            boundary_response=BOUNDARY_TEXT["out_of_year"].format(
                first=CORPUS_YEARS.start, last=CORPUS_YEARS.stop - 1, value=out_of_range[0]),
            reasons=(f"reading: {reading.origin}", f"year out of range: {out_of_range[0]}"),
        )

    change = reading.change
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

    program = reading.program
    program_codes = _PSHB_CODES if program == "PSHB" else _FEHB_CODES
    if program == "PSHB":
        reasons.append("PSHB program cue")
    plans = []
    for key in reading.options:
        if key in program_codes and program_codes[key] not in plans:
            # 'elevate plus' also matches the bare 'elevate' pattern; both
            # map to the same brochure, so dedupe handles it.
            plans.append(program_codes[key])
            reasons.append(f"plan cue: {key} -> {program_codes[key]}")

    as_of = reading.as_of
    if as_of:
        reasons.append(f"as of {as_of}")
    cover_field, cover_keys, cover_asked, cover_level = None, (), None, None
    hierarchy = (hierarchies or DEFAULT_HIERARCHIES).get("brochure", ())
    if program and not plans and "plan_code" in hierarchy:
        # The coverage rule: a level named (program), the level beneath it not
        # (plan) -> search every plan of that program and keep the best of each.
        cover_field, cover_asked, cover_level = "plan_code", program, "program"
        cover_keys = tuple(dict.fromkeys(program_codes.values()))
        plans = list(cover_keys)
        reasons.append(f"program named, no plan -> cover every {program} plan {cover_keys}")
    elif plans and not program and "program" in hierarchy and "plan_code" in hierarchy:
        # The same rule one level up: an option named (HDHP, High, Standard)
        # but not the program above it, and the option exists under more than
        # one program -> cover every plan offering it, in the routed years.
        # Never a guess at FEHB. A member's enrollment, when bound, narrows
        # this to their plan (retrieval.bind_enrollment_plan).
        offered = _plan_years()
        option_keys = list(reading.options)
        keys = tuple(dict.fromkeys(
            codes[key] for key in option_keys for codes in (_FEHB_CODES, _PSHB_CODES)
            if any(y in offered.get(codes[key], set()) for y in years)))
        if len(keys) > 1:
            label = " / ".join(dict.fromkeys(_OPTION_LABEL[k] for k in option_keys))
            cover_field, cover_asked, cover_level, cover_keys = "plan_code", label, "option", keys
            plans = list(keys)
            reasons.append(f"option named, no program -> cover every plan offering it {keys}")
    return Route(
        scope="in_scope",
        years=years,
        plan_codes=tuple(plans),
        reasons=tuple(reasons),
        as_of=as_of,
        change=change,
        cover_field=cover_field,
        cover_keys=cover_keys,
        cover_asked=cover_asked,
        cover_level=cover_level,
    )


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], start=1)}
_AS_OF = re.compile(
    r"\b(?:as of|in effect on|effective(?: on)?|on the date of service,?)\s+"
    r"(?P<d>(?:\d{4}-\d{2}-\d{2})|(?:\d{1,2}/\d{1,2}/\d{2,4})|(?:[A-Z][a-z]+\.? \d{1,2},? \d{4}))", re.IGNORECASE)


def as_of_date(query: str) -> str | None:
    """An explicit in-effect date in the question, as ISO; None otherwise."""
    m = _AS_OF.search(query)
    if not m:
        return None
    raw = m.group("d")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return raw
        if "/" in raw:
            mm, dd, yy = raw.split("/")
            year = int(yy) + (2000 if len(yy) == 2 else 0)
            return f"{year:04d}-{int(mm):02d}-{int(dd):02d}"
        month, day, year = raw.replace(",", "").replace(".", "").split()
        return f"{int(year):04d}-{_MONTHS[month.lower()[:9]] if month.lower() in _MONTHS else _MONTHS[[k for k in _MONTHS if k.startswith(month.lower()[:3])][0]]:02d}-{int(day):02d}"
    except (ValueError, IndexError, KeyError):
        return None


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
