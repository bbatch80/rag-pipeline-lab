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

import re
import uuid
from dataclasses import dataclass, field

import psycopg

from raglab import payload as payload_mod
from raglab import retrieval, router, snowlane
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
                "query_name": self.query_name, "slots": list(self.slots), "required": self.required}


@dataclass
class Plan:
    shape: str                     # simple | compound
    legs: list[Leg]
    origin: str                    # rules | caller | model
    model: str | None = None
    widened: bool = False

    def to_dict(self) -> dict:
        return {"shape": self.shape, "origin": self.origin, "model": self.model,
                "widened": self.widened, "legs": [leg.to_dict() for leg in self.legs]}


@dataclass
class Caller:
    """The identity every leg runs as. persona None = admin (owner connection)."""
    persona: str | None = None
    warehouse_role: str | None = None
    user_id: int | None = None


def menu(conn: psycopg.Connection) -> dict:
    """The fixed menu a plan may choose from: document source families that
    are ingested, and the named-query catalog. Nothing outside it executes."""
    return {"sources": tuple(retrieval._ingested_sources(conn)),
            "named_queries": tuple(snowlane.NAMED_QUERIES)}


def plan_rules(question: str) -> Plan:
    """The fast path: the v1 router fully routes a question as one document
    leg on the caller's original text. Simple by construction."""
    return Plan(shape="simple", origin="rules", legs=[Leg(name="documents", kind="doc_probe", text=question)])


def plan_from_dict(spec: dict, origin: str = "caller", model: str | None = None) -> Plan:
    legs = [Leg(name=l.get("name") or f"leg{i + 1}", kind=l["kind"], text=l.get("text"),
                sources=tuple(l.get("sources") or ()), query_name=l.get("query_name"),
                slots=tuple(l.get("slots") or ()), params=dict(l.get("params") or {}),
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
            bad = set(leg.params) - allowed
            if bad:
                raise PlanError(f"leg {leg.name}: parameters not accepted by {leg.query_name}: {sorted(bad)}")
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
    bound = dict(leg.params)
    for slot in leg.slots:
        if slot not in values:
            raise PlanError(f"leg {leg.name}: unknown slot {slot!r}")
        if values[slot] is None:
            raise PlanError(f"leg {leg.name}: {slot} required but not in context")
        bound[slot] = values[slot]
    return bound


def _member_id_of(ctx: retrieval.Context) -> str | None:
    return ctx.record.get("member_id")


def compose(
    conn: psycopg.Connection,
    question: str,
    caller: Caller,
    member_id: str | None = None,
    plan: Plan | None = None,
    source: str = "interactive",
    sf_connect=None,
) -> dict:
    """Plan -> execute every leg as the caller -> compose one payload ->
    disclose once. `plan=None` takes the rules fast path (P3-PR2 inserts the
    model between). `sf_connect` opens the warehouse session for the caller's
    role (injected so tests never touch Snowflake)."""
    if caller.persona is not None and caller.persona not in PERSONAS:
        raise ValueError(f"unknown persona {caller.persona!r}; expected one of {PERSONAS}")
    watch = Stopwatch()
    with watch.stage("resolve"):
        route = router.route(question)
        ctx = retrieval.resolve_context(conn, member_id, question) if route.scope == "in_scope" \
            else retrieval.Context(query=question)
        if member_id:
            ctx.record.setdefault("member_id", member_id)
    plan = plan or plan_rules(question)
    with watch.stage("plan"):
        validate(plan, question, menu(conn))

    sub_results: list[dict] = []
    warehouse_results: list[dict] = []
    chunks: list[dict] = []
    all_reranked: list = []
    as_of_defaulted = False
    for leg in plan.legs:
        if leg.kind == "doc_probe":
            result, reranked, widened = _run_doc_leg(conn, leg, question, caller, ctx, route, watch)
            plan.widened = plan.widened or widened
            start = len(chunks)
            built = payload_mod.build(leg.text, result.decision, reranked)
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
            warehouse_results.append(_run_member_leg(leg, caller, ctx, member_id, route, watch, sf_connect))

    with watch.stage("payload"):
        built = payload_mod.compose(question, plan.to_dict(), sub_results, warehouse_results, chunks,
                                    subject=member_id or _member_id_of(ctx), unresolved=ctx.unresolved,
                                    as_of_defaulted=as_of_defaulted)
        built["payload_id"] = str(uuid.uuid4())
        built["persona"] = caller.persona or "admin"
        built["member_context"] = member_id
        built["record_context"] = {**ctx.record, **({"as_of": ctx.as_of} if ctx.as_of else {})}
    _disclose_and_commit(conn, built, all_reranked, source, caller.user_id, watch)
    return built


def _touches_policies(leg: Leg) -> bool:
    return not leg.sources or "clinical_policy" in leg.sources


def _leg_route(leg: Leg, route: router.Route, ctx: retrieval.Context) -> router.Route:
    """The leg's route: the question's filters (years, plan, scope) with the
    leg's source hints and the bound as-of date (decision 5)."""
    from dataclasses import replace
    as_of = route.as_of or ctx.as_of
    return replace(route, sources=tuple(leg.sources) or route.sources, as_of=as_of,
                   reasons=route.reasons + ((f"as_of bound from the claim's date of service {as_of}",) if ctx.as_of and not route.as_of else ()))


def _run_doc_leg(conn, leg: Leg, question: str, caller: Caller, ctx: retrieval.Context, route: router.Route, watch: Stopwatch):
    leg_route = _leg_route(leg, route, ctx)
    probe = _probe(conn, leg.text, caller.persona, ctx, watch, decision=leg_route)
    insufficient, _ = payload_mod.abstention_verdict(probe.reranked) if probe.reranked else (True, None)
    if insufficient and leg.required and leg.sources:
        # Widen once: same question, same identity, same policies, hints removed.
        from dataclasses import replace
        probe = _probe(conn, leg.text, caller.persona, ctx, watch, decision=replace(leg_route, sources=route.sources))
        return probe, probe.reranked, True
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
        sf = connect(role=caller.warehouse_role)
        try:
            result = snowlane.run_named_query(sf, leg.query_name, params)
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
            "masked_columns": result["masked_columns"]}
