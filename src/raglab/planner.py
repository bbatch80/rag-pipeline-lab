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

MAX_LEGS = 3
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

    def to_dict(self) -> dict:
        return {"shape": self.shape, "origin": self.origin, "model": self.model, "module": self.module, "widened": self.widened,
                "fallback_reason": self.fallback_reason, "legs": [leg.to_dict() for leg in self.legs]}


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
MEMBER_QUERIES = ("member_calls", "member_recent_claims", "member_claims_summary", "member_enrollment",
                  "claim_adjudication", "member_denials", "provider_lookup", "provider_network_status", "providers_by_specialty")
MODULES = {
    "ask": {"sources": BENEFITS_DOCS, "named_queries": ()},
    "agent_assist": {"sources": ("call_note",) + BENEFITS_DOCS, "named_queries": MEMBER_QUERIES},
    "appeals_workbench": {"sources": ("appeal", "call_note", "clinical_note") + BENEFITS_DOCS,
                          "named_queries": ("appeal_case", "member_appeals") + MEMBER_QUERIES},
    "care_management": {"sources": ("clinical_note", "clinical_policy", "brochure", "formulary"),
                        "named_queries": ("member_claims_summary", "member_recent_claims", "member_enrollment")},
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
    return Plan(shape=spec.get("shape") or ("compound" if len(legs) > 1 else "simple"), legs=legs, origin=origin, model=model)


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
PLANNER_TIMEOUT_S = float(os.environ.get("RAGLAB_PLANNER_TIMEOUT", "8"))

# The catalog's free-text parameters (never identifiers, never names): the
# closed set a plan's `params` object may carry.
FREE_TEXT_PARAMS = tuple(sorted({
    param for spec in snowlane.NAMED_QUERIES.values() for param in spec.get("params", {})
} - set(SLOT_NAMES) - {"last_name", "first_name", "name"}))

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


def plan_with_model(conn: psycopg.Connection, question: str, available: dict, client=None) -> Plan:
    """Shape + legs from the pinned model in one constrained call — stored
    plan first (decision 3), the live model on a miss, the rules plan on any
    failure (timeout, invalid JSON, a plan that fails validation) with the
    reason recorded. Never raises for model trouble."""
    translated = translated_for_planning(conn, question)
    key = (_hash(translated), menu_hash(available), PLANNER_MODEL)
    if PLAN_CACHE:
        row = _stored_plan(conn, key)
        if row is not None:
            plan = plan_from_dict(row["plan"], origin=row["origin"], model=PLANNER_MODEL)
            plan.stored = True
            return plan
    t0 = time.perf_counter()
    reason = None
    try:
        raw = _call_model(translated, available, client)
        plan = plan_from_dict(raw, origin="model", model=PLANNER_MODEL)
        validate(plan, question, available)
    except Exception as exc:  # noqa: BLE001 — every model failure degrades to rules, and says why
        reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        plan = plan_rules(question)
        plan.model = PLANNER_MODEL
    plan.fallback_reason = reason
    if PLAN_CACHE:
        _store_plan(conn, key, translated, plan, reason, (time.perf_counter() - t0) * 1000)
    return plan


def _call_model(translated: str, available: dict, client=None) -> dict:
    import anthropic

    client = client or anthropic.Anthropic(timeout=PLANNER_TIMEOUT_S, max_retries=1)
    response = client.messages.create(
        model=PLANNER_MODEL,
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
    stored = _stored_plan(conn, (_hash(translated), menu_hash(available), PLANNER_MODEL))
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
) -> dict:
    """Plan -> execute every leg as the caller -> compose one payload ->
    disclose once. `plan=None` takes the rules fast path (P3-PR2 inserts the
    model between). `sf_connect` opens the warehouse session for the caller's
    role (injected so tests never touch Snowflake)."""
    if caller.persona is not None and caller.persona not in PERSONAS:
        raise ValueError(f"unknown persona {caller.persona!r}; expected one of {PERSONAS}")
    watch = Stopwatch()
    with watch.stage("resolve"):
        from raglab.pipeline import _hierarchies, coverage_note
        route = router.route(question, hierarchies=_hierarchies(conn))
        ctx = retrieval.resolve_context(conn, member_id, question) if route.scope == "in_scope" \
            else retrieval.Context(query=question)
        if member_id:
            ctx.record.setdefault("member_id", member_id)
    _t(trace, "question", text=question, persona=caller.persona or "admin", warehouse_role=caller.warehouse_role,
       member_id=member_id, module=module)
    _t(trace, "route", scope=route.scope, years=list(route.years), plan_codes=list(route.plan_codes), as_of=route.as_of,
       reasons=list(route.reasons))
    _t(trace, "identifiers", member_key=ctx.member_key, record=dict(ctx.record), as_of=ctx.as_of,
       unresolved=list(ctx.unresolved))
    available = menu(conn, module)
    _t(trace, "menu", module=module, sources=list(available["sources"]), named_queries=list(available["named_queries"]))
    if route.scope == "in_scope" and plan is None:
        _t(trace, "translated_for_planner", text=translated_for_planning(conn, question) if PLANNER == "model" else None)
    plan = plan or (plan_for(conn, question, module=module) if route.scope == "in_scope" else plan_rules(question))
    with watch.stage("plan"):
        validate(plan, question, available)
        plan.module = module
    _t(trace, "plan", origin=plan.origin, stored=plan.stored, model=plan.model, shape=plan.shape,
       fallback_reason=plan.fallback_reason, legs=[leg.to_dict() for leg in plan.legs])

    sub_results: list[dict] = []
    warehouse_results: list[dict] = []
    chunks: list[dict] = []
    all_reranked: list = []
    as_of_defaulted = False
    coverage = None
    for leg in plan.legs:
        if leg.kind == "doc_probe":
            result, reranked, widened = _run_doc_leg(conn, leg, question, caller, ctx, route, watch, available, trace)
            plan.widened = plan.widened or widened
            start = len(chunks)
            built = payload_mod.build(leg.text, result.decision, reranked, coverage=coverage_note(conn, result.decision, reranked))
            coverage = coverage or built.get("coverage")
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
                                    as_of_defaulted=as_of_defaulted, coverage=coverage)
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
    return replace(route, sources=sources, as_of=as_of,
                   reasons=route.reasons + ((f"as_of bound from the claim's date of service {as_of}",) if ctx.as_of and not route.as_of else ()))


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


def _run_doc_leg(conn, leg: Leg, question: str, caller: Caller, ctx: retrieval.Context, route: router.Route, watch: Stopwatch,
                 available: dict | None = None, trace: list | None = None):
    """One document leg over the module's document menu (or every visible
    source when unscoped). Widen-once is retired with decision 7: nothing is
    filtered by a hint, so there is nothing to widen."""
    leg_route = _leg_route(leg, route, ctx, available)
    text = _leg_text(conn, leg, ctx)
    probe = _probe(conn, text, caller.persona, ctx, watch, decision=leg_route)
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
           persona=caller.persona or "admin", sources_searched=list(leg_route.sources), hints=list(leg.sources),
           filters={"years": list(leg_route.years), "plan_codes": list(leg_route.plan_codes), "as_of": leg_route.as_of,
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
