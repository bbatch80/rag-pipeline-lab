"""Composed context (Phase 3): one question -> one plan of up to three legs
-> one payload, one disclosure, one status.

A plan is a fixed-menu object: document probes (the v1 funnel, ranked on the
leg's text) and member queries (the Snowflake catalog by name). The planner
holds no privileges: every leg executes as the caller's identity — the
Postgres persona for documents, the warehouse role for named queries.

Identifiers never come from a plan. The member key, record keys, and the
claim's date of service are resolved from the ORIGINAL question (shape +
check digit + lookup) and bound to every leg by the platform; a leg whose
text carries an identifier the question does not is rejected before
anything runs (decision 4). Plans are one pass (D14): no leg is decided from
another's result. Widen-once: a required document leg that came back
insufficient with source hints is re-run once without them, same identity.

Origins: `rules` (the v1 router — one document leg on the original text; the
fast path), `caller` (a plan supplied by the caller, e.g. a golden item),
`model` (P3-PR2: the pinned planner model on the translated question)."""
from __future__ import annotations

import hashlib
import difflib
import functools
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field

import psycopg

from raglab import payload as payload_mod
from raglab import rerank, retrieval, router, snowlane
from raglab.pipeline import PERSONAS, _disclose_and_commit, _probe
from raglab.timing import Stopwatch

MAX_LEGS = 4  # 2026-09-14: one fact-shaped leg per listed benefit (four topics in the user's probe)
LEG_KINDS = ("doc_probe", "member_query")
_ID_TOKEN = retrieval._ID_TOKEN


class PlanError(ValueError):
    """A plan that must not execute: unknown menu item, too many legs, an
    identifier the question does not contain, a slot the context can't fill."""


@dataclass
class Leg:
    name: str
    kind: str                      # doc_probe | member_query
    text: str | None = None        # doc_probe: the sub-question ranked within this leg
    sources: tuple[str, ...] = ()  # doc_probe: doc_type hints (empty = every visible source)
    query_name: str | None = None  # member_query: a catalog name
    slots: tuple[str, ...] = ()    # member_query: which context values bind (member_id, claim_id, case_id, ...)
    params: dict = field(default_factory=dict)  # member_query: validated free-text catalog params only
    required: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "text": self.text, "sources": list(self.sources),
                "query_name": self.query_name, "slots": list(self.slots), "params": dict(self.params), "required": self.required}


@dataclass
class Plan:
    shape: str                     # simple | compound
    legs: list[Leg]
    origin: str                    # rules | caller | model
    model: str | None = None
    widened: bool = False
    fallback_reason: str | None = None
    module: str | None = None
    stored: bool = False  # reused from the plans table (no model call this time)
    enforced: tuple[str, ...] = ()  # rules the platform applied on top of the model's plan (e.g. document_leg)

    def to_dict(self) -> dict:
        return {"shape": self.shape, "origin": self.origin, "model": self.model, "module": self.module, "widened": self.widened,
                "fallback_reason": self.fallback_reason, "enforced": list(self.enforced), "legs": [leg.to_dict() for leg in self.legs]}


@dataclass
class Caller:
    """The identity every leg runs as. persona None = admin (owner connection)."""
    persona: str | None = None
    warehouse_role: str | None = None
    user_id: int | None = None


# Job-shaped modules over the unified layer (Phase 3 decision 7): each
# interface gets a fixed menu of document families and named queries. The
# planner chooses ONLY from the module's menu; the caller's entitlement
# (RLS, warehouse grants) trims it further at execution. module=None is the
# unscoped menu — the MCP experiment and the eval's shape metric, never a
# production surface.
BENEFITS_DOCS = ("brochure", "rates", "sop", "bulletin", "formulary", "kb", "clinical_policy", "carrier_letter")
MEMBER_QUERIES = ("member_profile", "member_calls", "member_recent_claims", "member_claims_summary", "member_enrollment",
                  "claim_adjudication", "member_denials", "provider_lookup", "provider_network_status", "providers_by_specialty")
MODULES = {
    "ask": {"sources": BENEFITS_DOCS, "named_queries": ()},
    "agent_assist": {"sources": ("call_note",) + BENEFITS_DOCS, "named_queries": ("member_appeals",) + MEMBER_QUERIES},
    "appeals_workbench": {"sources": ("appeal", "call_note", "clinical_note") + BENEFITS_DOCS,
                          "named_queries": ("appeal_case", "member_appeals") + MEMBER_QUERIES},
    "care_management": {"sources": ("clinical_note", "clinical_policy", "brochure", "formulary"),
                        "named_queries": ("member_profile", "member_claims_summary", "member_recent_claims", "member_enrollment", "member_appeals")},
    "analyst_view": {"sources": ("brochure", "rates", "clinical_policy", "carrier_letter", "formulary"),
                     "named_queries": ("cost_by_condition", "providers_by_specialty", "provider_network_status")},
}


# What each document family CONTAINS, for the planner's menu — the registry
# names a source and its department; the model needs to know which questions
# the text answers and which the warehouse answers instead. Part of the menu
# hash: edit a line and stored plans recompute.
SOURCE_GUIDE = {
    "brochure": "FEHB/PSHB plan brochures: what a plan covers, copays, deductibles, exclusions, how to file claims and appeals — by plan and year.",
    "rates": "Premium rate tables by plan, option, and enrollment type.",
    "sop": "Member-services standard operating procedures: how reps handle situations.",
    "bulletin": "Claims bulletins: internal instructions to examiners.",
    "formulary": "Drug formulary: tiers and coverage of medications.",
    "kb": "CSR knowledge base articles.",
    "clinical_note": "Clinical notes about one patient (SOAP, referral, discharge): diagnoses, treatment, what the clinician wrote.",
    "call_note": "Call notes: the rep's written narrative of one member's call — what the member asked or disputed, what they were told, next steps. (The warehouse call log holds only when a call happened and its reason code.)",
    "appeal": "Appeal case documents: the case summary, denial rationale, clinical summary of submitted records, determination letters. (The warehouse appeals table holds only the case's dates, decision, and reviewer.)",
    "clinical_policy": "Medical policies: coverage criteria for procedures and services, versioned by effective date.",
    "carrier_letter": "OPM carrier letters: program-wide instructions to carriers by year.",
}


def menu(conn: psycopg.Connection, module: str | None = None) -> dict:
    """The fixed menu a plan may choose from: document source families that
    are ingested (with the registry's one-line description, so the model can
    tell call NOTES — what was said — from the call LOG in the warehouse),
    and the named-query catalog. Nothing outside it executes. Descriptions
    are part of the menu hash: change one and stored plans recompute."""
    ingested = tuple(retrieval._ingested_sources(conn))
    try:
        with conn.transaction():
            rows = conn.execute("SELECT doc_type, display_name, department FROM sources WHERE doc_type = ANY(%s)", (list(ingested),)).fetchall()
    except psycopg.Error:
        rows = []
    registry = {dt: f"{name} ({dept})" for dt, name, dept in rows}
    sources, queries = ingested, tuple(snowlane.NAMED_QUERIES)
    if module is not None:
        if module not in MODULES:
            raise PlanError(f"unknown module {module!r}; expected one of {sorted(MODULES)}")
        sources = tuple(dt for dt in ingested if dt in MODULES[module]["sources"])
        queries = tuple(q for q in MODULES[module]["named_queries"] if q in snowlane.NAMED_QUERIES)
    return {"module": module, "sources": sources, "named_queries": queries,
            "source_docs": {dt: SOURCE_GUIDE.get(dt, registry.get(dt, dt)) for dt in sources}}


def plan_rules(question: str) -> Plan:
    """The fast path: the v1 router fully routes a question as one document
    leg on the caller's original text. Simple by construction."""
    return Plan(shape="simple", origin="rules", legs=[Leg(name="documents", kind="doc_probe", text=question)])


def plan_from_dict(spec: dict, origin: str = "caller", model: str | None = None) -> Plan:
    legs = [Leg(name=l.get("name") or f"leg{i + 1}", kind=l["kind"], text=l.get("text"),
                sources=tuple(l.get("sources") or ()), query_name=l.get("query_name"),
                slots=tuple(l.get("slots") or ()), params={k: v for k, v in (l.get("params") or {}).items() if v is not None},
                required=bool(l.get("required", True)))
            for i, l in enumerate(spec.get("legs", []))]
    plan = Plan(shape=spec.get("shape") or ("compound" if len(legs) > 1 else "simple"), legs=legs, origin=origin, model=model)
    plan.enforced = tuple(spec.get("enforced") or ())
    return plan


# A question about what a document SAID needs a document leg. The prompt
# says so; at temperature 0 the model still answers "what did the letter
# tell the member" with the warehouse row alone (appeal-03,
# persona_negative-03, the user's Workbench probe 2026-09-14). The platform
# enforces it: when the question asks what was written and the plan carries
# no doc_probe, one is added over the module's whole document menu.
_DOCUMENT_WORDS = re.compile(
    r"\b(letter|said|says|say|state[sd]?|wrote|written|describe[sd]?|summari[sz]e|summary|tell|told|argue[sd]?|call(?:ed|s)? about|"
    r"rationale|reasoning|explain(?:ed|s)?|why|what (?:does|did|do) .{0,40}\b(?:say|state|require|cover|mean))\b",
    re.IGNORECASE,
)


# An open-ended benefits question — "how does X coverage work", "what does the
# plan cover for X", "tell me about the X benefit" — has no answer-shaped
# sentence for a cross-encoder to score (the mental-health rows scored 0.002
# against it, 0.99 against "what is the copay for X"). The platform expands
# such a document leg into fact-shaped legs by template: the benefit's cost
# row, its coverage terms, its limits. A rule, not a prompt change: it fires
# only on this shape and leaves every other plan untouched (a prompt version
# re-planned everything and lost nine golden items, run 667).
_OPEN_ENDED = re.compile(
    r"\bhow (?:does|do|is|are) (?:the |my |our |their )?(?P<a>[\w\- /&']+?) (?:coverage |benefits? )?(?:work|covered|handled)\b|"
    r"\bwhat does (?:the |my |this )?(?:plan |option )?cover for (?P<b>[\w\- /&']+?)\??$|"
    r"\btell me about (?:the |my )?(?P<c>[\w\- /&']+?) (?:benefits?|coverage)\b|"
    r"\bhow (?:is|are) (?P<d>[\w\- /&']+?) covered\b",
    re.IGNORECASE,
)
_BENEFIT_DOC_SOURCES = ("brochure", "clinical_policy", "formulary", "kb")


def fact_shaped_legs(topic: str, sources: tuple) -> list:
    topic = topic.strip().rstrip("?").strip()
    return [
        Leg(name=f"{topic}: cost", kind="doc_probe", text=f"What is the copay or coinsurance for {topic}?", sources=sources),
        Leg(name=f"{topic}: coverage", kind="doc_probe", text=f"What {topic} services are covered in-network and out-of-network?", sources=sources),
        Leg(name=f"{topic}: limits", kind="doc_probe", text=f"Are there visit limits, precertification requirements, or exclusions for {topic}?",
            sources=sources, required=False),
    ]


def expand_open_ended_legs(plan: Plan, question: str) -> Plan:
    """Replace an open-ended document leg with fact-shaped legs (see above).
    Never touches a caller's plan, a warehouse leg, a leg over records, or a
    plan that already has more than one document leg."""
    if plan.origin == "caller":
        return plan
    doc_legs = [leg for leg in plan.legs if leg.kind == "doc_probe"]
    if len(doc_legs) != 1:
        return plan
    leg = doc_legs[0]
    if leg.sources and not set(leg.sources) & set(_BENEFIT_DOC_SOURCES):
        return plan
    m = _OPEN_ENDED.search(question or "") or _OPEN_ENDED.search(leg.text or "")
    if not m:
        return plan
    topic = next(g for g in m.groups() if g)
    if len(topic.split()) > 6:
        return plan
    others = [l for l in plan.legs if l.kind != "doc_probe"]
    room = MAX_LEGS - len(others)
    if room < 2:
        return plan
    sources = tuple(s for s in leg.sources if s in _BENEFIT_DOC_SOURCES) or ()
    plan.legs = others + fact_shaped_legs(topic, sources)[:room]
    plan.shape = "compound" if len(plan.legs) > 1 else "simple"
    plan.enforced = tuple(plan.enforced) + ("fact_shaped_legs",)
    return plan


# A question that names SEVERAL benefits at once ("the costs for physical
# therapy, imaging, outpatient surgery, and specialist visits") asks one
# leg to find a chunk answering all of them — none does; the best partial
# match (a specialist-copay article) took the seats and the rest got
# nothing. Split it: one fact-shaped leg per named benefit, up to four.
_BENEFIT_LIST = re.compile(
    r"\b(?P<ask>costs?|copays?|coinsurance|coverage|benefits?|prices?|cost[- ]sharing)\s+(?:for|of|on)\s+(?P<list>[^?.]+?)(?:\s+(?:on|under|in|with)\s+(?:the\s+)?(?P<plan>[\w\- ]*?(?:option|plan|hdhp|elevate(?: plus)?)))?\s*\??$",
    re.IGNORECASE,
)
_COVER_LIST = re.compile(r"\b(?:does|do) (?:the |my )?(?:plan|option|coverage)? ?cover\s+(?P<list>[^?.]+?)\s*\??$", re.IGNORECASE)


def _split_list(text: str) -> list[str]:
    parts = [p.strip(" ,;") for p in re.split(r",|\band\b|\bor\b|;|/", text) if p and p.strip(" ,;")]
    parts = [p for p in parts if 1 <= len(p.split()) <= 4]
    return parts if 2 <= len(parts) <= MAX_LEGS else []


_TITLES_SYSTEM = ("You know how FEHB and PSHB plan brochures title the rows of their Section 5 benefits tables "
                  "(for example 'Lab, x-ray and other diagnostic tests', 'Physical, occupational, speech, habilitative and rehabilitative therapy', "
                  "'Outpatient hospital or ambulatory surgical center'). For each benefit a member names, answer with the brochure row title "
                  "it falls under, in the same order, as a JSON array of strings and nothing else.")


@functools.lru_cache(maxsize=256)
def benefit_row_titles(items: tuple[str, ...], client=None) -> tuple[str, ...] | None:
    """The brochure's own row title for each named benefit, from the planner
    model at temperature 0; None when the call fails or answers badly (the
    leg then carries the member's words). The reranker scores a row against
    its own title near 1 and against a synonym near 0 (2026-09-14)."""
    try:
        import anthropic

        client = client or anthropic.Anthropic(timeout=PLANNER_TIMEOUT_S, max_retries=1)
        response = client.messages.create(
            model=PLANNER_MODEL, extra_body={"temperature": PLANNER_TEMPERATURE}, max_tokens=300, system=_TITLES_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(list(items))}])
        text = next(b.text for b in response.content if b.type == "text")
        titles = json.loads(text[text.index("["):text.rindex("]") + 1])
    except Exception:  # noqa: BLE001 — the member's words still make a leg
        return None
    if not (isinstance(titles, list) and len(titles) == len(items) and all(isinstance(t, str) and 0 < len(t) <= 120 for t in titles)):
        return None
    return tuple(t.strip() for t in titles)


def split_benefit_list(plan: Plan, question: str, titles=None) -> Plan:
    """One fact-shaped document leg per named benefit. Only for a plan with
    exactly one document leg (over a benefits family or unhinted) that no
    other rule has already reshaped; never a caller's plan."""
    if plan.origin == "caller" or "fact_shaped_legs" in plan.enforced:
        return plan
    doc_legs = [leg for leg in plan.legs if leg.kind == "doc_probe"]
    if len(doc_legs) != 1:
        return plan
    leg = doc_legs[0]
    if leg.sources and not set(leg.sources) & set(_BENEFIT_DOC_SOURCES):
        return plan
    text = question or leg.text or ""
    m = _BENEFIT_LIST.search(text)
    items, plan_words, ask = [], "", "cost"
    if m:
        items, plan_words, ask = _split_list(m.group("list")), (m.group("plan") or "").strip(), m.group("ask").lower()
    else:
        m = _COVER_LIST.search(text)
        if m:
            items, ask = _split_list(m.group("list")), "cover"
    if not items:
        return plan
    others = [l for l in plan.legs if l.kind != "doc_probe"]
    if len(others) + len(items) > MAX_LEGS:
        return plan
    suffix = f" on the {plan_words}" if plan_words else ""
    sources = tuple(s for s in leg.sources if s in _BENEFIT_DOC_SOURCES) or ()
    rows = (titles if titles is not None else benefit_row_titles)(tuple(items)) or tuple(items)
    if ask == "cover":
        legs = [Leg(name=f"{it}: coverage", kind="doc_probe", text=f"Does the plan cover {row}{suffix}?", sources=sources) for it, row in zip(items, rows)]
    else:
        legs = [Leg(name=f"{it}: cost", kind="doc_probe", text=f"What do I pay for {row}{suffix}?", sources=sources) for it, row in zip(items, rows)]
    plan.legs = others + legs
    plan.shape = "compound" if len(plan.legs) > 1 else "simple"
    plan.enforced = tuple(plan.enforced) + ("split_by_benefit",)
    return plan


def enforce_document_leg(plan: Plan, question: str) -> Plan:
    """Add a document leg to a plan that has none when the question asks what
    a document said. Never touches a caller's plan, a plan already at the
    leg limit, or a plan that already searches documents."""
    if plan.origin == "caller" or len(plan.legs) >= MAX_LEGS:
        return plan
    if any(leg.kind == "doc_probe" for leg in plan.legs):
        return plan
    if not _DOCUMENT_WORDS.search(question or ""):
        return plan
    plan.legs.append(Leg(name="documents", kind="doc_probe", text=question, sources=()))
    plan.shape = "compound" if len(plan.legs) > 1 else "simple"
    plan.enforced = tuple(plan.enforced) + ("document_leg",)
    return plan


def fill_member_slot(plan: Plan) -> Plan:
    """A member-scoped query without its member is meaningless: when the
    catalog declares member_id and the leg neither binds the slot nor sets the
    parameter, bind the slot from context (the user's live probe 2026-09-14:
    member_appeals ran with no member and answered 'no rows')."""
    notes = []
    for leg in plan.legs:
        if leg.kind == "doc_probe":
            continue
        declared = snowlane.NAMED_QUERIES.get(leg.query_name, {}).get("params", {})
        if "member_id" in declared and "member_id" not in leg.slots and "member_id" not in leg.params:
            leg.slots = tuple(leg.slots) + ("member_id",)
            notes.append(f"member_slot:{leg.query_name}")
    if notes:
        plan.enforced = tuple(plan.enforced) + tuple(notes)
    return plan


def repair_query_names(plan: Plan, available: dict) -> Plan:
    """The model sometimes invents a query name for a real menu entry
    ('member_appeals_history' for member_appeals). Snap it to the closest name
    on the menu and say so; a name nothing resembles loses only its own leg.
    Before this, one invented name discarded the whole plan for a rules plan
    (the user's live probe 2026-09-14)."""
    menu = list(available.get("named_queries", ()))
    kept, notes = [], []
    for leg in plan.legs:
        if leg.kind == "doc_probe" or leg.query_name in menu:
            kept.append(leg)
            continue
        close = difflib.get_close_matches(leg.query_name or "", menu, n=1, cutoff=0.6)
        if close:
            notes.append(f"query_name:{leg.query_name}->{close[0]}")
            leg.query_name = close[0]
            kept.append(leg)
        else:
            notes.append(f"dropped:{leg.query_name}")
    if notes:
        plan.legs = kept
        plan.shape = "compound" if len(plan.legs) > 1 else "simple"
        plan.enforced = tuple(plan.enforced) + tuple(notes)
    return plan


def validate(plan: Plan, question: str, available: dict) -> None:
    """Schema-check before anything executes."""
    if not 1 <= len(plan.legs) <= MAX_LEGS:
        raise PlanError(f"a plan has 1..{MAX_LEGS} legs, got {len(plan.legs)}")
    if plan.shape not in ("simple", "compound"):
        raise PlanError(f"unknown shape {plan.shape!r}")
    if len({leg.name for leg in plan.legs}) != len(plan.legs):
        raise PlanError("leg names must be unique")
    question_ids = {_canon(m.group(0)) for m in _ID_TOKEN.finditer(question)}
    for leg in plan.legs:
        if leg.kind not in LEG_KINDS:
            raise PlanError(f"leg {leg.name}: unknown kind {leg.kind!r}")
        if leg.kind == "doc_probe":
            if not leg.text:
                raise PlanError(f"leg {leg.name}: a document probe needs text")
            unknown = set(leg.sources) - set(available["sources"])
            if unknown:
                raise PlanError(f"leg {leg.name}: sources not on the menu: {sorted(unknown)}")
            foreign = {_canon(m.group(0)) for m in _ID_TOKEN.finditer(leg.text)} - question_ids
            if foreign:
                raise PlanError(f"leg {leg.name}: identifier not in the question: {sorted(foreign)}")
        else:
            if leg.query_name not in available["named_queries"]:
                raise PlanError(f"leg {leg.name}: named query not in the catalog: {leg.query_name!r}")
            allowed = set(snowlane.NAMED_QUERIES[leg.query_name].get("params", {}))
            bad_slots = set(leg.slots) - allowed
            if bad_slots:
                raise PlanError(f"leg {leg.name}: {leg.query_name} takes no {sorted(bad_slots)} parameter")
            bad = set(leg.params) - allowed
            if bad:
                raise PlanError(f"leg {leg.name}: parameters not accepted by {leg.query_name}: {sorted(bad)}")
            if set(leg.params) & {"last_name", "first_name", "name"}:
                raise PlanError(f"leg {leg.name}: a person's name is never a query parameter (names never resolve)")
            for k, v in leg.params.items():
                if k in SLOT_NAMES or re.search(r"\d{7,}", str(v)):
                    raise PlanError(f"leg {leg.name}: {k} is bound from context, never written into a plan")


SLOT_NAMES = ("member_id", "claim_id", "case_id", "npi", "plan_code", "as_of")


def _canon(token: str) -> str:
    return re.sub(r"[\s-]", "", token).upper()


def bind_slots(leg: Leg, ctx: retrieval.Context, member_id: str | None, route: router.Route) -> dict:
    """Fill a member query's parameters from resolved context only."""
    values = {"member_id": member_id or _member_id_of(ctx), "claim_id": ctx.record.get("claim_id"),
              "case_id": ctx.record.get("case_id"), "plan_code": (route.plan_codes or (None,))[0],
              "as_of": ctx.as_of}
    # Free-text params arrive as strings from a plan; the catalog declares each
    # parameter's type (limit is int) — cast to it, or the SQL fails to compile.
    declared = snowlane.NAMED_QUERIES.get(leg.query_name, {}).get("params", {})
    bound = {}
    for k, v in leg.params.items():
        t = declared.get(k)
        if t is int:
            try:
                v = int(str(v).strip())
            except ValueError as exc:
                raise PlanError(f"leg {leg.name}: {k} must be an integer, got {v!r}") from exc
        bound[k] = v
    for slot in leg.slots:
        if slot not in values:
            raise PlanError(f"leg {leg.name}: unknown slot {slot!r}")
        if values[slot] is None:
            raise PlanError(f"leg {leg.name}: {slot} required but not in context")
        bound[slot] = values[slot]
    return bound


def _member_id_of(ctx: retrieval.Context) -> str | None:
    return ctx.record.get("member_id")


# ---- the pinned planner model (Phase 3 decisions 2-4) --------------------

PLANNER = os.environ.get("RAGLAB_PLANNER", "model")          # model | rules
PLAN_CACHE = os.environ.get("RAGLAB_PLAN_CACHE", "on") != "off"
PLANNER_MODEL = os.environ.get("RAGLAB_PLANNER_MODEL", "claude-haiku-4-5-20251001")  # pinned: the key of every stored plan
# Deterministic planning (2026-09-14): both model calls sample at temperature
# 0, so the same question gets the same plan and a menu change re-plans only
# where the new option matters. Measured before: fresh plans at the default
# temperature flipped about one golden item in ten. The sampling setting is
# part of the store key, so plans made the old way are never reused.
PLANNER_TEMPERATURE = 0.0


def plan_key_model() -> str:
    """The model component of a store key: the pinned model plus its sampling."""
    return f"{PLANNER_MODEL}@t{PLANNER_TEMPERATURE:g}"
PLANNER_TIMEOUT_S = float(os.environ.get("RAGLAB_PLANNER_TIMEOUT", "8"))

# The catalog's free-text parameters (never identifiers, never names): the
# closed set a plan's `params` object may carry.
FREE_TEXT_PARAMS = tuple(sorted({
    param for spec in snowlane.NAMED_QUERIES.values() for param in spec.get("params", {})
} - set(SLOT_NAMES) - {"last_name", "first_name", "name"}))

READER_SYSTEM = """You read one question asked of a governed retrieval platform for a health insurer (GEHA). You never answer it, never search, and never see records. Report only what the question SAYS, as fields the platform enforces:
- `scope`: "other_carrier" ONLY when the question asks for a fact ABOUT another insurer's plan (Blue Cross / FEP, Aetna, Kaiser, MHBP, NALC, APWU, Cigna, Humana, Anthem …) — what it charges, covers, or requires, or a comparison with it — and put that name in `boundary_value`; without a name it is not a boundary. A carrier named only as the asker's former or other plan, as a contrast, or as background is NOT the subject: "Unlike my old Aetna plan, do I need a referral with GEHA High?" is in scope (it asks about GEHA); "I'm switching from Kaiser; does GEHA cover my prescriptions?" is in scope; "What does Blue Cross FEP charge for a specialist?" is other_carrier; "How does GEHA Standard compare to Blue Cross Standard?" is other_carrier (the comparison needs Blue Cross facts). The words "carrier", "carriers", "examiners", "OPM", claims bulletins, appeals, and notes are all GEHA's own business and are in scope. "medicare_program" when it asks for Medicare's OWN program facts (Medicare premiums, Part B or Part D amounts, IRMAA) rather than how a GEHA plan coordinates with Medicare or what GEHA reimburses. Otherwise "in_scope" with `boundary_value` null.
- `program`: "FEHB" only when the question literally says FEHB or federal employees; "PSHB" only when it says PSHB or postal. Otherwise "none". Never infer the program from an option or a product name: High, Standard, HDHP and "GEHA Benefit Plan" exist under both programs.
- `options`: the plan options the question names, as keys: "hdhp" (HDHP, high-deductible), "elevate", "elevate_plus", "high" (High Option, "hi opt", the GEHA Benefit Plan), "standard" (Standard Option, "std"). [] when none.
- `years`: the plan years written in the question (e.g. 2026); [] when none. Never infer a year.
- `change`: true when the question asks how something changed or compares years (changed, compare, difference, increase, vs).
- `as_of`: an explicit in-effect date written in the question, as YYYY-MM-DD ("as of March 1, 2025", "in effect on 2026-01-06"); otherwise null.
Identifiers appear as tokens like [MEMBER_ID-12]; they carry no meaning for these fields. Return only the JSON object."""

ROUTE_SCHEMA = {  # the reading of the route (2026-09-12): every value from a registry-declared set
    "type": "object", "additionalProperties": False,
    "required": ["scope", "boundary_value", "program", "options", "years", "change", "as_of"],
    "properties": {
        "scope": {"type": "string", "enum": ["in_scope", "other_carrier", "medicare_program"]},
        "boundary_value": {"type": ["string", "null"]},
        "program": {"type": "string", "enum": ["FEHB", "PSHB", "none"]},  # constrained decoding: an enum cannot be nullable
        "options": {"type": "array", "items": {"type": "string", "enum": ["hdhp", "elevate", "elevate_plus", "high", "standard"]}},
        "years": {"type": "array", "items": {"type": "integer"}},
        "change": {"type": "boolean"},
        "as_of": {"type": ["string", "null"]},
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shape", "legs"],
    "properties": {
        "shape": {"type": "string", "enum": ["simple", "compound"]},
        "legs": {  # the API's constrained decoding rejects minItems/maxItems: validate() enforces 1..MAX_LEGS
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "kind", "text", "sources", "query_name", "slots", "params", "required"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(LEG_KINDS)},
                    "text": {"type": ["string", "null"]},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "query_name": {"type": ["string", "null"]},
                    "slots": {"type": "array", "items": {"type": "string", "enum": list(SLOT_NAMES)}},
                    "params": {"type": "object", "additionalProperties": False,  # the API's constrained decoding needs a closed object
                               "properties": {name: {"type": ["string", "null"]} for name in FREE_TEXT_PARAMS}},
                    "required": {"type": "boolean"},
                },
            },
        },
    },
}

SYSTEM = """You plan how a governed retrieval platform for a health insurer answers one question. You never answer the question and never see any records.

Decide the SHAPE: "simple" when one search or one catalog query answers it; "compound" when it needs more than one (at most three).

Each leg is one of:
- doc_probe: a search over documents. Give `text` — the sub-question this leg should rank on, written in the question's own words and tokens. `sources` is a list of document families to search (from the menu), or [] for every family the caller may see. `query_name` null, `slots` [].
- member_query: one named warehouse query from the catalog (menu). Give `query_name`. `slots` lists which identifiers from the question it needs: member_id, claim_id, case_id, npi, plan_code, as_of. `params` carries the query's FREE-TEXT parameters only (e.g. description_like as an ILIKE pattern like '%asthma%', speciality, zip_prefix, limit) — never an identifier. `text` null, `sources` [].

Rules:
- Identifiers appear as tokens like [MEMBER_ID-12], [CLAIM_ID-7], [CASE_ID-3], [PERSON-5], [DATE_TIME-9]. Never invent, alter, or expand a token; copy tokens exactly when a leg's text needs them. You never write identifier values — the platform binds them.
- The warehouse catalog answers questions about facts in records: a claim's status or denial, costs, enrollment, providers, network status, when calls or appeals happened and how they were decided. Documents answer questions about what was WRITTEN: what a brochure or policy says, what a rep or clinician wrote, what an appeal file argues. "What did the member say/dispute/ask" is a call_note document; "when did the member call" is the call log.
- `required` is true when the question cannot be answered without that leg; false for supporting context.
- Prefer the fewest legs that cover the question. Never plan a leg the question did not ask for: "summarize / what does X say / what was written" is ONE doc_probe leg — do not add enrollment, claims, or call-log legs for background. Use member_query only when the question asks for a fact the catalog holds.
- `slots` may list only identifier kinds the question actually contains AND that the named query accepts (its parameters are listed in the menu). Never add a slot the query does not take.
Return only the JSON object."""


def _menu_text(available: dict) -> str:
    docs = available.get("source_docs") or {dt: dt for dt in available["sources"]}
    families = "\n".join(f"  - {dt}: {desc}" for dt, desc in docs.items())
    catalog = "\n".join(
        f"  - {name} (slots: {', '.join(sorted(set(snowlane.NAMED_QUERIES[name].get('params', {})) & set(SLOT_NAMES)) or ['none'])}; "
        f"free-text params: {', '.join(sorted(set(snowlane.NAMED_QUERIES[name].get('params', {})) - set(SLOT_NAMES) - {'last_name', 'first_name', 'name'}) or ['none'])}): "
        f"{snowlane.NAMED_QUERIES[name]['doc']}"
        for name in available["named_queries"] if name in snowlane.NAMED_QUERIES)
    return f"Document families (doc_probe sources):\n{families}\nNamed queries (member_query query_name):\n{catalog}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def menu_hash(available: dict) -> str:
    return _hash(json.dumps({"module": available.get("module"), "sources": sorted(available["sources"]), "named_queries": sorted(available["named_queries"]),
                             "source_docs": dict(sorted((available.get("source_docs") or {}).items())),
                             "system": SYSTEM}, sort_keys=True))  # the instructions shape the plan too: edit them and plans recompute


def translated_for_planning(conn: psycopg.Connection, question: str) -> str:
    """What the model sees: the question with identifiers and names replaced
    by vault tokens (decision 4). Planning runs on the owner connection
    before any role drop, so the full translation is available; a token
    tells the model the KIND of identifier, never the value."""
    from raglab import deid
    return deid.translate_query(conn, question)


def _plan_prepare(conn: psycopg.Connection, question: str, available: dict) -> tuple[tuple, str, Plan | None]:
    """The database half before the model: the translated question, the store
    key, and the stored plan if there is one."""
    translated = translated_for_planning(conn, question)
    key = (_hash(translated), menu_hash(available), plan_key_model())
    if PLAN_CACHE:
        row = _stored_plan(conn, key)
        if row is not None:
            plan = plan_from_dict(row["plan"], origin=row["origin"], model=PLANNER_MODEL)
            plan.stored = True
            return key, translated, plan
    return key, translated, None


def _plan_finish(conn: psycopg.Connection, question: str, available: dict, key: tuple, translated: str,
                 outcome, latency_ms: float) -> Plan:
    """The database half after the model: `outcome` is the raw plan dict or
    the exception the call raised. Any failure degrades to rules with the
    reason recorded; the result is stored either way."""
    reason = None
    try:
        if isinstance(outcome, BaseException):
            raise outcome
        plan = plan_from_dict(outcome, origin="model", model=PLANNER_MODEL)
        plan = repair_query_names(plan, available)
        validate(plan, question, available)
        plan = split_benefit_list(plan, question)  # the titled legs are stored with the plan: CI and the VM reuse them without a model call
    except Exception as exc:  # noqa: BLE001 — every model failure degrades to rules, and says why
        reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        plan = plan_rules(question)
        plan.model = PLANNER_MODEL
    plan.fallback_reason = reason
    if PLAN_CACHE:
        _store_plan(conn, key, translated, plan, reason, latency_ms)
    return plan


def _guarded(fn, *args):
    """Run a model call and return its result or the exception (never raise):
    the finish step decides what a failure means."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        return exc


def plan_with_model(conn: psycopg.Connection, question: str, available: dict, client=None) -> Plan:
    """Shape + legs from the pinned model in one constrained call — stored
    plan first (decision 3), the live model on a miss, the rules plan on any
    failure (timeout, invalid JSON, a plan that fails validation) with the
    reason recorded. Never raises for model trouble."""
    key, translated, stored = _plan_prepare(conn, question, available)
    if stored is not None:
        return stored
    t0 = time.perf_counter()
    outcome = _guarded(_call_model, translated, available, client)
    return _plan_finish(conn, question, available, key, translated, outcome, (time.perf_counter() - t0) * 1000)


def _call_model(translated: str, available: dict, client=None) -> dict:
    import anthropic

    client = client or anthropic.Anthropic(timeout=PLANNER_TIMEOUT_S, max_retries=1)
    response = client.messages.create(
        model=PLANNER_MODEL,
        extra_body={"temperature": PLANNER_TEMPERATURE},  # SDK 1.0 dropped the parameter; the endpoint still honors it
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Menu:\n{_menu_text(available)}\n\nQuestion: {translated}"}],
        output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _stored_plan(conn, key) -> dict | None:
    """Only a plan the MODEL produced is reused; a rules fallback is logged in
    the same table but never answers for the model — otherwise one transient
    timeout would pin a rules plan to that question for good."""
    try:
        with conn.transaction():
            row = conn.execute("SELECT plan, origin FROM plans WHERE question_hash = %s AND menu_hash = %s AND model = %s "
                               "AND origin = 'model'", key).fetchone()
    except psycopg.Error:
        return None
    return {"plan": row[0], "origin": row[1]} if row else None


def _store_plan(conn, key, translated: str, plan: Plan, reason: str | None, latency_ms: float) -> None:
    try:
        with conn.transaction():
            conn.execute(
                "INSERT INTO plans (question_hash, menu_hash, model, question, shape, origin, plan, fallback_reason, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (question_hash, menu_hash, model) DO UPDATE SET shape = EXCLUDED.shape, origin = EXCLUDED.origin, "
                "plan = EXCLUDED.plan, fallback_reason = EXCLUDED.fallback_reason, latency_ms = EXCLUDED.latency_ms, created_at = now() "
                "WHERE plans.origin <> 'model'",  # a model plan is never overwritten by a later fallback
                (*key, translated, plan.shape, plan.origin, json.dumps(plan.to_dict()), reason, round(latency_ms, 1)))
    except psycopg.Error:
        pass  # the store is an accelerator and a log, never a gate on answering


READER_MENU_HASH = "reader-v1"  # the reader has no menu; this versions its prompt in the store key


def _reading_from_dict(r: dict) -> router.Reading:
    program = r.get("program")
    return router.Reading(program=program if program in ("FEHB", "PSHB") else None, options=tuple(r.get("options") or ()),
                          years=tuple(int(y) for y in (r.get("years") or ())), change=bool(r.get("change")),
                          as_of=r.get("as_of") or None, scope=r.get("scope") or "in_scope",
                          boundary_value=r.get("boundary_value") or None, origin="model")


def _validate_reading(r: dict) -> None:
    years = r.get("years") or []
    if not all(isinstance(y, int) and 2000 <= y <= 2100 for y in years):
        raise PlanError(f"reading names impossible years {years}")
    if r.get("as_of") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r["as_of"])):
        raise PlanError(f"reading as_of is not an ISO date: {r['as_of']!r}")


def _call_reader(translated: str, client=None) -> dict:
    import anthropic

    client = client or anthropic.Anthropic(timeout=PLANNER_TIMEOUT_S, max_retries=1)
    response = client.messages.create(
        model=PLANNER_MODEL, extra_body={"temperature": PLANNER_TEMPERATURE}, max_tokens=200, system=READER_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {translated}"}],
        output_config={"format": {"type": "json_schema", "schema": ROUTE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _read_prepare(conn: psycopg.Connection, question: str) -> tuple[tuple, str, router.Reading | None]:
    translated = translated_for_planning(conn, question)
    key = (_hash(translated), READER_MENU_HASH + ":" + _hash(READER_SYSTEM)[:12], plan_key_model())
    if PLAN_CACHE:
        row = _stored_plan(conn, key)
        if row is not None:
            return key, translated, _reading_from_dict(row["plan"])
    return key, translated, None


def _read_finish(conn: psycopg.Connection, question: str, key: tuple, translated: str, outcome, latency_ms: float) -> router.Reading:
    """`outcome` is the raw reading or the exception the call raised; any
    failure degrades to the regex reader, logged and never stored as the model's."""
    try:
        if isinstance(outcome, BaseException):
            raise outcome
        raw = outcome
        _validate_reading(raw)
        reading = _reading_from_dict(raw)
        stored = Plan(shape="reading", legs=[], origin="model", model=PLANNER_MODEL)
        stored.to_dict = lambda: raw  # the store holds the raw reading
        reason = None
    except Exception as exc:  # noqa: BLE001 — every model failure degrades to the regex reader, and says why
        reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        reading = router.read(question)
        stored = Plan(shape="reading", legs=[], origin="rules", model=PLANNER_MODEL)
        stored.to_dict = lambda: {"origin": "rules"}
    if PLAN_CACHE:
        _store_plan(conn, key, translated, stored, reason, latency_ms)
    return reading


def read_with_model(conn: psycopg.Connection, question: str, client=None) -> router.Reading:
    """The reading of the route from the pinned model in its own constrained
    call (2026-09-12, two calls: reading and planning are different judgments
    with different gates). Stored reading first; the regex reader on any
    failure, with the reason logged and never stored as the model's."""
    key, translated, stored = _read_prepare(conn, question)
    if stored is not None:
        return stored
    t0 = time.perf_counter()
    outcome = _guarded(_call_reader, translated, client)
    return _read_finish(conn, question, key, translated, outcome, (time.perf_counter() - t0) * 1000)


def read_and_plan(conn: psycopg.Connection, question: str, available: dict, want_plan: bool,
                  client=None) -> tuple[router.Reading, "Plan | None"]:
    """The reading and the plan together: store lookups first; whatever missed
    is asked of the model CONCURRENTLY (two independent network calls, ~2-3 s
    each — sequentially they were the larger part of a fresh question's
    latency); every database step stays on this thread. `want_plan=False`
    (the rules planner, or a caller-supplied plan) asks for the reading only.
    A question the reading then routes out of scope wastes one planner call;
    its plan is discarded unstored, as before."""
    if PLANNER != "model":
        return router.read(question), (plan_rules(question) if want_plan else None)
    r_key, r_text, reading = _read_prepare(conn, question)
    p_key, p_text, plan = (None, None, None)
    if want_plan:
        p_key, p_text, plan = _plan_prepare(conn, question, available)
    need_read, need_plan = reading is None, want_plan and plan is None
    if need_read and need_plan:
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_read = pool.submit(_guarded, _call_reader, r_text, client)
            f_plan = pool.submit(_guarded, _call_model, p_text, available, client)
            read_out, plan_out = f_read.result(), f_plan.result()
        ms = (time.perf_counter() - t0) * 1000
        reading = _read_finish(conn, question, r_key, r_text, read_out, ms)
        plan = _PendingPlan(p_key, p_text, plan_out, ms)  # finished by the caller once the route is known
    elif need_read:
        t0 = time.perf_counter()
        reading = _read_finish(conn, question, r_key, r_text, _guarded(_call_reader, r_text, client),
                               (time.perf_counter() - t0) * 1000)
    elif need_plan:
        t0 = time.perf_counter()
        plan = _PendingPlan(p_key, p_text, _guarded(_call_model, p_text, available, client), (time.perf_counter() - t0) * 1000)
    return reading, plan


@dataclass
class _PendingPlan:
    """A planner outcome not yet validated or stored — the route decides
    whether it is needed (an out-of-scope question runs no plan)."""
    key: tuple
    translated: str
    outcome: object
    latency_ms: float

    def finish(self, conn, question: str, available: dict) -> Plan:
        return _plan_finish(conn, question, available, self.key, self.translated, self.outcome, self.latency_ms)


def read_route(conn: psycopg.Connection, question: str, client=None) -> router.Reading:
    """The reading the platform routes on: the model's (stored first) by
    configuration, else the regex reader's."""
    if PLANNER == "model":
        return read_with_model(conn, question, client)
    return router.read(question)


def plan_for(conn: psycopg.Connection, question: str, client=None, module: str | None = None) -> Plan:
    """The plan the platform would execute for a question, by configuration:
    the model (stored plan first) over the module's menu, or the rules fast path."""
    if PLANNER == "model":
        return plan_with_model(conn, question, menu(conn, module), client)
    return plan_rules(question)


def replan(conn: psycopg.Connection, question: str, client=None, module: str | None = None) -> tuple[dict | None, Plan]:
    """Ask the live model fresh and return (stored plan or None, fresh plan)
    without touching the store — the non-gating drift report (decision 3)."""
    available = menu(conn, module)
    translated = translated_for_planning(conn, question)
    stored = _stored_plan(conn, (_hash(translated), menu_hash(available), plan_key_model()))
    try:
        fresh = plan_from_dict(_call_model(translated, available, client), origin="model", model=PLANNER_MODEL)
        validate(fresh, question, available)
    except Exception as exc:  # noqa: BLE001
        fresh = plan_rules(question); fresh.fallback_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
    return (stored["plan"] if stored else None), fresh


def compose(
    conn: psycopg.Connection,
    question: str,
    caller: Caller,
    member_id: str | None = None,
    plan: Plan | None = None,
    source: str = "interactive",
    sf_connect=None,
    module: str | None = None,
    trace: list | None = None,
    case_id: str | None = None,
) -> dict:
    """Plan -> execute every leg as the caller -> compose one payload ->
    disclose once. `plan=None` takes the rules fast path (P3-PR2 inserts the
    model between). `sf_connect` opens the warehouse session for the caller's
    role (injected so tests never touch Snowflake)."""
    if caller.persona is not None and caller.persona not in PERSONAS:
        raise ValueError(f"unknown persona {caller.persona!r}; expected one of {PERSONAS}")
    watch = Stopwatch()
    available = menu(conn, module)
    caller_plan = plan is not None
    with watch.stage("read"):
        # The model READS the route (stored reading first); the code enforces
        # it. The plan's model call, when needed, runs concurrently with the
        # reading's and is finished below once the route is known.
        reading, pending = read_and_plan(conn, question, available, want_plan=not caller_plan)
    with watch.stage("resolve"):
        from raglab.pipeline import _hierarchies, coverage_note
        route = router.route(question, hierarchies=_hierarchies(conn), reading=reading)
        ctx = retrieval.resolve_context(conn, member_id, question, case_id=case_id) if route.scope == "in_scope" \
            else retrieval.Context(query=question)
        if member_id:
            ctx.record.setdefault("member_id", member_id)
        # The member's enrolled plan binds ONCE here, so warehouse legs (slot
        # plan_code) and document legs (plan filter) both see it. Probe 6: it
        # was bound inside the document step only, and 'which doctors are in
        # network for this member' was refused for want of a plan code.
        route = retrieval.bind_enrollment_plan(route, ctx)
    _t(trace, "question", text=question, persona=caller.persona or "admin", warehouse_role=caller.warehouse_role,
       member_id=member_id, case_id=case_id, module=module)
    _t(trace, "route", scope=route.scope, years=list(route.years), plan_codes=list(route.plan_codes), as_of=route.as_of,
       reasons=list(route.reasons), reading={"origin": reading.origin, "program": reading.program, "options": list(reading.options),
                                             "years": list(reading.years), "change": reading.change, "as_of": reading.as_of,
                                             "scope": reading.scope})
    _t(trace, "identifiers", member_key=ctx.member_key, record=dict(ctx.record), as_of=ctx.as_of,
       unresolved=list(ctx.unresolved))
    _t(trace, "menu", module=module, sources=list(available["sources"]), named_queries=list(available["named_queries"]))
    if route.scope != "in_scope":
        # An out-of-scope route means NO retrieval: no plan, no legs. The payload
        # says out_of_scope with the boundary text at the top, as the single-
        # query path always has. Before 2026-09-12 the composer ran the rules
        # plan, its one leg reported out_of_scope, and the top-level status
        # became 'insufficient_evidence: missing documents' with the boundary
        # text lost (scope_negative-01: a Blue Cross question).
        with watch.stage("payload"):
            built = payload_mod.build(question, route, [])
            built["payload_id"] = str(uuid.uuid4())
            built["persona"] = caller.persona or "admin"
            built["member_context"] = member_id
            built["record_context"] = dict(ctx.record)
            built["unresolved_identifiers"] = list(ctx.unresolved)
            built["plan"] = None
            built["sub_results"], built["warehouse_results"], built["missing"] = [], [], []
        _t(trace, "composed", status=built["status"], boundary=built.get("boundary_response"), chunks=0)
        _disclose_and_commit(conn, built, [], source, caller.user_id, watch)
        return built
    if not caller_plan:
        _t(trace, "translated_for_planner", text=translated_for_planning(conn, question) if PLANNER == "model" else None)
    with watch.stage("plan"):
        if plan is None:
            plan = pending.finish(conn, question, available) if isinstance(pending, _PendingPlan) else pending
        validate(plan, question, available)
        plan = fill_member_slot(plan)
        plan = enforce_document_leg(plan, question)
        plan = expand_open_ended_legs(plan, question)
        plan = split_benefit_list(plan, question)
        plan.module = module
    _t(trace, "plan", origin=plan.origin, stored=plan.stored, model=plan.model, shape=plan.shape,
       fallback_reason=plan.fallback_reason, legs=[leg.to_dict() for leg in plan.legs])

    sub_results: list[dict] = []
    warehouse_results: list[dict] = []
    chunks: list[dict] = []
    all_reranked: list = []
    as_of_defaulted = False
    coverage = None
    search_note: dict = {}
    for leg in plan.legs:
        if leg.kind == "doc_probe":
            result, reranked, widened = _run_doc_leg(conn, leg, question, caller, ctx, route, watch, available, trace)
            plan.widened = plan.widened or widened
            start = len(chunks)
            built = payload_mod.build(leg.text, result.decision, reranked, coverage=coverage_note(conn, result.decision, reranked),
                                      search=result.search_stats, identity_evidence=result.identity_evidence)
            if result.identity_evidence and "member_records" not in plan.enforced:
                plan.enforced = tuple(plan.enforced) + ("member_records",)
            coverage = coverage or built.get("coverage")
            for k in ("readings", "candidates", "trimmed"):  # summed over the document legs
                if k in result.search_stats:
                    search_note[k] = search_note.get(k, 0) + result.search_stats[k]
            if "cap" in result.search_stats:
                search_note["cap"] = result.search_stats["cap"]
            _t(trace, "leg_verdict", leg=leg.name, status=built["status"], confidence=built.get("confidence"),
               thresholds={"prose": rerank.ABSTAIN_THRESHOLD, **rerank.ABSTAIN_BY_SOURCE})
            for c in built["chunks"]:
                c["leg"] = leg.name
            chunks.extend(built["chunks"])
            all_reranked.extend(reranked[: len(built["chunks"])])
            if _touches_policies(leg) and ctx.as_of is None and route.as_of is None:
                as_of_defaulted = True
            sub_results.append({"leg": leg.name, "status": built["status"], "confidence": built.get("confidence"),
                                "router": built.get("router", {}), "chunk_indexes": list(range(start, len(chunks))),
                                "widened": widened, "reason": None if built["status"] == "ok" else built["status"]})
        else:
            w = _run_member_leg(leg, caller, ctx, member_id, route, watch, sf_connect)
            warehouse_results.append(w)
            _t(trace, "warehouse_leg", leg=leg.name, query_name=leg.query_name, role=caller.warehouse_role,
               status=w["status"], reason=w.get("reason"), row_count=w.get("row_count"), columns=w.get("columns"),
               rows=(w.get("rows") or [])[:3], masked_columns=w.get("masked_columns"), bound=w.get("bound"))

    with watch.stage("payload"):
        built = payload_mod.compose(question, plan.to_dict(), sub_results, warehouse_results, chunks,
                                    subject=member_id or _member_id_of(ctx), unresolved=ctx.unresolved,
                                    as_of_defaulted=as_of_defaulted, coverage=coverage, search=search_note or None,
                                    router={"years": list(route.years), "plan_codes": list(route.plan_codes), "as_of": route.as_of,
                                            "plan_from_enrollment": route.plan_from_enrollment})
        built["payload_id"] = str(uuid.uuid4())
        built["persona"] = caller.persona or "admin"
        built["member_context"] = member_id
        built["record_context"] = {**ctx.record, **({"as_of": ctx.as_of} if ctx.as_of else {})}
    _t(trace, "composed", status=built["status"], missing=list(built["missing"]), chunks=len(chunks),
       confidence=built.get("confidence"), as_of_defaulted=as_of_defaulted, subject=built["subject"], coverage=coverage)
    _disclose_and_commit(conn, built, all_reranked, source, caller.user_id, watch)
    _t(trace, "disclosed", payload_id=built["payload_id"], source=source, timings=built.get("timings"))
    return built


def _t(trace: list | None, stage: str, **data) -> None:
    """Record one stage of a composition for `raglab trace` — nothing is
    computed for the trace that the pipeline did not compute anyway."""
    if trace is not None:
        trace.append({"stage": stage, **data})


def _touches_policies(leg: Leg) -> bool:
    return not leg.sources or "clinical_policy" in leg.sources


def _leg_route(leg: Leg, route: router.Route, ctx: retrieval.Context, available: dict | None = None) -> router.Route:
    """The leg's route: the question's filters (years, plan, scope), the
    MODULE's whole document menu as the sources searched (decision 7: a
    hint is recorded in the plan, never applied as a filter — a model guess
    must not hide a source), and the bound as-of date (decision 5)."""
    from dataclasses import replace
    as_of = route.as_of or ctx.as_of
    sources = tuple(available["sources"]) if available and available.get("module") else route.sources
    leg_route = replace(route, sources=sources, as_of=as_of,
                        reasons=route.reasons + ((f"as_of bound from the claim's date of service {as_of}",) if ctx.as_of and not route.as_of else ()))
    if route.cover_level == "all" and leg.sources and "brochure" not in leg.sources:
        # "Every plan" coverage exists for brochure questions; a leg the
        # planner aimed at a bulletin, the formulary, or a policy must not
        # search six plans' brochures beside it (run 669: three internal-
        # document items lost their seats to brochure look-alikes).
        leg_route = replace(leg_route, plan_codes=(), cover_field=None, cover_keys=(), cover_asked=None, cover_level=None,
                            reasons=leg_route.reasons + (f"leg aimed at {list(leg.sources)}: every-plan coverage not applied",))
    return leg_route


def _leg_text(conn, leg: Leg, ctx: retrieval.Context) -> str:
    """The leg's ranking text — with the applied policy's title appended when
    the question named a case or claim (record context, never the model) and
    this leg may reach policies: the Phase 1 thin-question mechanism."""
    text = leg.text or ""
    if ctx.record.get("policy_id") and "clinical_policy" in leg.sources:  # an EXPLICIT policy leg only
        title = retrieval.policy_title(conn, ctx.record["policy_id"])
        if title and title.lower() not in text.lower():
            text = f"{text} ({title})"
    return text


# An open-ended request for a member's records — the records are the answer.
# A fact question over the records ("do the notes say what was prescribed?")
# keeps the score-based verdict: the records are seated, but the platform
# says insufficient when none of them carries the fact (unanswerable-05).
_RECORDS_REQUEST = re.compile(
    r"\b(history|summar(?:y|i[sz]e)|describe|overview|background|tell me about|what do we know|"
    r"everything|all (?:of )?(?:the |their |his |her )?(?:calls|notes|records|visits)|"
    r"recent (?:calls|notes|visits)|any (?:calls|notes|records)|what (?:calls|notes|records) (?:are|do|does)|"
    r"what (?:was|were|is|are) (?:done|said|written|noted|discussed)|"
    r"what (?:did|do|does) (?:they|she|he|the member|the rep|the nurse|the doctor|it) (?:call|ask|say|dispute|write|tell|report)|"
    r"what (?:was|were|is|are) (?:it|this|that|(?:the|their|his|her) (?:\w+ ){0,3}\w+) (?:about|for|regarding))\b",
    re.IGNORECASE,
)
# "...say WHAT the doctor prescribed": a specific fact asked of the records —
# the records are seated, but the verdict stays with the scores.
_NESTED_FACT = re.compile(r"\b(?:say|says|said|mention|mentions|state|states|note|notes|indicate|show)s? (?:what|whether|if|which|how much|how many|when|where|who)\b",
                          re.IGNORECASE)


def apply_records_rule(reranked: list, candidates: list, hinted: set, member_key: str | None, top_n: int,
                       open_ended: bool = True) -> tuple[list, int]:
    """The records rule (2026-09-14): when a member is open and the planner
    aimed this leg at a member-scoped source, the member's own records in
    that source ARE the evidence — seated first in reranker order and never
    abstained on for scoring low against a question they were not written
    to answer ("tell me about this member's clinical history": her discharge
    summary scored 0.004 and lost to policy boilerplate). Other sources fill
    the remaining seats under their normal bar. Returns (seated, how many
    were seated by identity)."""
    if not member_key or not hinted:
        return reranked, 0
    pool = candidates or reranked
    records = sorted((c for c in pool if c.doc_type in hinted), key=lambda c: c.rerank_score or 0.0, reverse=True)
    if not records:
        return reranked, 0
    seated = records[:top_n]
    ids = {c.chunk_id for c in seated}
    for c in reranked:
        if len(seated) >= top_n:
            break
        if c.chunk_id not in ids:
            seated.append(c)
            ids.add(c.chunk_id)
    return seated, (len(records[:top_n]) if open_ended else 0)


def _run_doc_leg(conn, leg: Leg, question: str, caller: Caller, ctx: retrieval.Context, route: router.Route, watch: Stopwatch,
                 available: dict | None = None, trace: list | None = None):
    """One document leg over the module's document menu (or every visible
    source when unscoped). Widen-once is retired with decision 7: nothing is
    filtered by a hint, so there is nothing to widen."""
    leg_route = _leg_route(leg, route, ctx, available)
    text = _leg_text(conn, leg, ctx)
    probe = _probe(conn, text, caller.persona, ctx, watch, decision=leg_route)
    if ctx.member_key and leg.sources:
        member_scoped = set(retrieval._source_flags(conn)[0])
        hinted = set(leg.sources) & member_scoped
        asked = f"{question or ''} {text or ''}"
        open_ended = bool(_RECORDS_REQUEST.search(asked)) and not _NESTED_FACT.search(asked)
        probe.reranked, probe.identity_evidence = apply_records_rule(probe.reranked, probe.candidates, hinted, ctx.member_key,
                                                                     rerank.TOP_N_OUT, open_ended=open_ended)
    if trace is not None:
        pool = probe.candidates or []
        def line(c):
            return {"title": c.doc_title, "doc_type": c.doc_type, "section": c.section, "vector_rank": c.vector_rank,
                    "text_rank": c.text_rank, "rrf": round(c.rrf_score, 4), "rerank": None if c.rerank_score is None else round(c.rerank_score, 4),
                    "text": (c.index_text or c.content)[:160]}
        by_vec = sorted([c for c in pool if c.vector_rank], key=lambda c: c.vector_rank)[:5]
        by_txt = sorted([c for c in pool if c.text_rank], key=lambda c: c.text_rank)[:5]
        fused = sorted(pool, key=lambda c: -c.rrf_score)[:5]
        _t(trace, "doc_leg", leg=leg.name, leg_text=leg.text, ranking_text=text, search_text=probe.search_query,
           persona=caller.persona or "admin", sources_searched=list(probe.decision.sources), hints=list(leg.sources),
           filters={"years": list(probe.decision.years), "plan_codes": list(probe.decision.plan_codes), "as_of": probe.decision.as_of,
                    "plan_from_enrollment": probe.decision.plan_from_enrollment,
                    "member_key": ctx.member_key, "record": dict(ctx.record)},
           pool_size=len(pool), pool_by_source={dt: sum(1 for c in pool if c.doc_type == dt) for dt in sorted({c.doc_type for c in pool})},
           vector_top=[line(c) for c in by_vec], bm25_top=[line(c) for c in by_txt], fused_top=[line(c) for c in fused],
           rerank_top=[line(c) for c in probe.reranked[:5]])
    return probe, probe.reranked, False


def _run_member_leg(leg: Leg, caller: Caller, ctx: retrieval.Context, member_id: str | None, route: router.Route,
                    watch: Stopwatch, sf_connect) -> dict:
    base = {"leg": leg.name, "query_name": leg.query_name}
    if caller.warehouse_role is None:
        return {**base, "status": "not_executed", "reason": "no warehouse identity for this caller"}
    try:
        params = bind_slots(leg, ctx, member_id, route)
    except PlanError as exc:
        return {**base, "status": "not_executed", "reason": str(exc)}
    connect = sf_connect or snowlane.connect
    with watch.stage(f"warehouse:{leg.name}"):
        try:
            sf = connect(role=caller.warehouse_role)
        except Exception as exc:  # noqa: BLE001 — the warehouse is unreachable: the leg is missing, the payload still composes
            return {**base, "status": "not_executed", "reason": f"warehouse unavailable: {type(exc).__name__}"}
        try:
            result = snowlane.run_named_query(sf, leg.query_name, params)
        except Exception as exc:  # noqa: BLE001 — e.g. the role has no grant on a table: the engine refused, the leg is missing
            return {**base, "status": "not_executed", "reason": f"refused for this role: {type(exc).__name__}: {str(exc)[:120]}"}
        finally:
            try:
                sf.close()
            except Exception:
                pass
    if result.get("status") != "ok":
        return {**base, "status": result.get("status", "not_executed"), "reason": "catalog refused the query"}
    return {**base, "status": "ok" if result["row_count"] else "insufficient_evidence",
            "reason": None if result["row_count"] else "no rows",
            "columns": result["columns"], "rows": result["rows"], "row_count": result["row_count"],
            "masked_columns": result["masked_columns"], "bound": params}
