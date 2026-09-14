"""Tier-1 evaluation: deterministic checks over the golden set, on the
composed path.

Every golden item runs through `planner.compose` as its own identity
(persona + warehouse role, member on the screen or typed in the question)
and is scored by the checks it declares — `expected_values`, `expected_set`,
`expected_legs`, `expected_rows`, `expected_coverage`, `exclude_sources`,
... (see `raglab.checks`). An item that declares nothing falls back to
hit@5 on its `sources`. hit@5, precision@5 and source_coverage are reported
for every item that lists sources, whether or not they gate.

Guardrail shapes: a persona wall composes the question as the entitled
persona (must answer and cite) and as each denied persona (must hold none of
the protected documents); an adversarial item composes the question and its
control and compares the route; an unanswerable item expects a status or a
trigger (scope gate, unresolved or invalid identifier).

Every run writes one row per item per check to `eval_scores`, an
`item_pass` verdict per item, and a `payload_health` row (top-two gap,
duplicates, index seats, needless leg splits). The summary reports pass
rate per work category and per group with Wilson intervals, the health
numbers, and a paired diff against the previous run.

The gate is a RATCHET against `eval/baseline.json` (`ratchet`): a run fails
when any category's or group's pass rate falls below the stored rate, or
when a guardrail item that passed at the baseline fails. Nothing gates on
an absolute bar. `write_baseline` stores a run as the reference.
"""

import hashlib
import json
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

import psycopg

from raglab import ablation, checks, config, deid, planner, rerank, retrieval, router, sources, stats, taxonomy
from raglab.mcp_server import IDENTITIES
from raglab.timing import Stopwatch, percentile

EVAL_SCHEMA_PATH = config.REPO_ROOT / "db" / "eval.sql"
BASELINE_PATH = config.REPO_ROOT / "eval" / "baseline.json"
WAREHOUSE_SKIPPED = "warehouse_skipped"


@dataclass
class RetrievalEvalResult:
    run_id: int
    by_category: dict = field(default_factory=dict)   # group -> metric means (reported)
    by_source: dict = field(default_factory=dict)
    by_work: dict = field(default_factory=dict)       # work category number -> {name, n, passed, unverified, rate, ci, failing}
    by_group: dict = field(default_factory=dict)      # group -> same shape
    overall: dict = field(default_factory=dict)
    overall_ci: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)        # tie rate, margin, duplicate share, index seats, needless splits
    diff: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    diff_against: int | None = None
    failures: list = field(default_factory=list)      # ratchet breaches (only with a baseline)
    replan: list = field(default_factory=list)
    corpus_hash: str = ""
    guardrails_passing: list = field(default_factory=list)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=config.REPO_ROOT, timeout=5,
        ).stdout.strip()
    except OSError:
        return ""


def expected_source(item: dict, registry) -> str:
    """The doc_type(s) a golden item's expected evidence lives in — what the
    per-source slice reports on. Brochure sources are (plan_code, year);
    internal sources are repo-relative paths under a source's directory."""
    types = set()
    for source in item.get("sources", []):
        if "plan_code" in source:
            types.add("brochure")
        elif "internal" in source:
            for rel in ("data/internal/" + source["internal"], "data/raw/" + source["internal"]):
                for s in registry.all:
                    if s.dir and rel.startswith(s.dir + "/") and s.doc_type:
                        types.add(s.doc_type)
    return "+".join(sorted(types)) if types else "none"


def corpus_hash(conn: psycopg.Connection) -> str:
    """Digest of the corpus as ingested: sha256 over every document's
    content_hash in source_path order. Changes when any document is added,
    removed, or re-ingested under a different recipe; stable otherwise."""
    row = conn.execute(
        "SELECT string_agg(content_hash, ',' ORDER BY source_path) FROM documents"
    ).fetchone()
    return hashlib.sha256((row[0] or "").encode()).hexdigest()


# ---------------------------------------------------------------- identities
def caller_for(item: dict) -> planner.Caller:
    """The identity an item runs as: its persona through the MCP identity
    table (persona -> document persona + warehouse role); no persona = admin."""
    persona = item.get("persona")
    if persona is None:
        return planner.Caller()
    doc_persona, role = IDENTITIES[persona]
    return planner.Caller(persona=doc_persona, warehouse_role=role)


def _only_the_warehouse_is_missing(payload: dict) -> bool:
    """insufficient_evidence caused solely by warehouse legs that could not
    run for want of a warehouse, while every document leg found evidence."""
    if payload.get("status") != "insufficient_evidence" or not checks._warehouse_unavailable(payload):
        return False
    not_run = {w.get("leg") for w in payload.get("warehouse_results") or [] if w.get("status") == "not_executed"}
    missing = set(payload.get("missing") or [])
    doc_legs = payload.get("sub_results") or []
    return bool(missing) and missing <= not_run and bool(doc_legs) and all(s.get("status") == "ok" for s in doc_legs)


def _member_on_screen(item: dict) -> str | None:
    """The member the screen has open — unless the item types the id into
    the question on purpose (`typed_member_id`), which tests that path."""
    return None if item.get("typed_member_id") else item.get("member_id")


def _compose(conn, item: dict, question: str, persona: str | None = None) -> dict:
    caller = caller_for({**item, "persona": persona}) if persona is not None else caller_for(item)
    return planner.compose(conn, question, caller, member_id=_member_on_screen(item), case_id=item.get("case_id"),
                           module=item.get("module"), source="eval")


# ---------------------------------------------------------------- one item
def score_item(conn, item: dict) -> list[tuple]:
    """(metric, value, detail) rows for one golden item on the composed path."""
    qid, category = item["id"], item["category"]
    rows: list[tuple] = []

    def add(metric, value, detail=None):
        rows.append((metric, float(value), detail or {}))

    # Malformed / unknown identifiers never reach a search: the resolver raises.
    trigger = item.get("expected_trigger")
    try:
        payload = _compose(conn, item, item["question"])
    except ValueError as exc:
        add("check_expected_trigger", trigger == "invalid_identifier", {"error": str(exc)[:160], "expected": trigger})
        return rows
    if trigger == "invalid_identifier":
        add("check_expected_trigger", 0.0, {"expected": trigger, "status": payload.get("status"), "note": "no error raised"})
    elif trigger == "unresolved_identifier":
        add("check_expected_trigger", bool(payload.get("unresolved_identifiers")) and payload.get("status") != "ok",
            {"unresolved": payload.get("unresolved_identifiers"), "status": payload.get("status")})

    # Reported retrieval metrics on the item's sources, then the declared checks.
    for metric, value, detail in checks.source_metrics(item, payload):
        add(metric, value, detail)
    declared = checks.declared_checks(item)
    warehouse_off = checks._warehouse_unavailable(payload)
    for metric, value, detail in checks.run_declared(item, payload, conn):
        field_name = metric[len(checks.CHECK_PREFIX):]
        if warehouse_off and field_name in checks.WAREHOUSE_CHECKS:
            add(WAREHOUSE_SKIPPED, 1.0, {"check": metric, "reason": "warehouse unavailable"})
        else:
            add(metric, value, detail)
    if warehouse_off and any(l.get("kind") == "member_query" for l in (payload.get("plan") or {}).get("legs") or []) \
            and "expected_legs" not in item and not declared:
        add(WAREHOUSE_SKIPPED, 1.0, {"check": "compose", "reason": "warehouse unavailable"})

    # Guardrail shapes.
    if "persona_allow" in item or "persona_deny" in item:
        if item.get("persona_allow"):
            allowed = _compose(conn, item, item["question"], persona=item["persona_allow"])
            if _only_the_warehouse_is_missing(allowed):
                # CI holds no warehouse: a required warehouse leg could not run,
                # so the status reads insufficient although the document legs
                # found the evidence. The wall is about the documents — judge
                # them, and record that the warehouse half was skipped.
                add(WAREHOUSE_SKIPPED, 1.0, {"check": "check_allow", "reason": "warehouse unavailable: required warehouse leg not executed"})
                allowed = {**allowed, "status": "ok"}
            add(*checks.check_allow_titles(item, allowed))
            for metric, value, detail in checks.source_metrics(item, allowed):
                add(metric, value, detail)
            payload = allowed
        for persona in (item.get("persona_deny"), item.get("persona_deny_2")):
            if persona:
                denied = _compose(conn, item, item["question"], persona=persona)
                add(*checks.check_deny_titles(item, denied, persona))
    if item.get("control_question") and ("expect_only_plan" in item or "expect_only_member" in item):
        # Route invariance: the instruction in the question must not move the
        # plan / year filters (items that test a different layer skip this).
        control = _compose(conn, item, item["control_question"])
        same = (sorted((payload.get("router") or {}).get("plan_codes") or []) == sorted((control.get("router") or {}).get("plan_codes") or [])
                and sorted((payload.get("router") or {}).get("years") or []) == sorted((control.get("router") or {}).get("years") or []))
        add("check_control_route", same, {"route": payload.get("router"), "control": control.get("router")})

    # Nothing declared and nothing guardrail-shaped: hit@5 is the check.
    gated = [m for m, *_ in rows if m.startswith(checks.CHECK_PREFIX) or m == WAREHOUSE_SKIPPED]
    if not gated:
        hit = next((v for m, v, _ in rows if m == "hit@5"), None)
        if hit is not None:
            add("check_hit", hit, {"fallback": "no expectation declared: hit@5 gates"})
        else:
            add("check_hit", 0.0, {"fallback": "no expectation and no sources: nothing to judge"})

    # Health and timings.
    add("payload_health", 1.0, checks.payload_health(payload))
    snap = payload.get("timings") or {}
    for stage in ("embed", "search", "rerank", "total"):
        if stage in snap:
            add(f"latency_{stage}", snap[stage], {"host": snap.get("host")})
    return rows


# ---------------------------------------------------------------- the run
def _score_items_parallel(items: list[dict], workers: int) -> list[list[tuple]]:
    """score_item over `items` on `workers` threads, each with its own owner
    connection (a scored item SETs a role and commits its disclosure — never
    on a shared connection). Results come back in golden order, so a parallel
    run stores exactly what a sequential run would. Model calls and embedding
    waits overlap; reranking is serialized inside rerank._predict."""
    import queue
    from concurrent.futures import ThreadPoolExecutor

    from raglab import db, rerank

    rerank._get_model()  # load once on this thread, not racing inside the pool
    pool: queue.Queue = queue.Queue()
    conns = [db.connect() for _ in range(workers)]
    for c in conns:
        pool.put(c)

    def one(item: dict) -> list[tuple]:
        c = pool.get()
        try:
            return score_item(c, item)
        finally:
            pool.put(c)

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval") as ex:
            return list(ex.map(one, items))
    finally:
        for c in conns:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


def run(
    conn: psycopg.Connection,
    config_label: str = "baseline",
    sabotage: bool = False,
    workers: int = 1,
    items_only: tuple[str, ...] = (),
    categories: tuple[str, ...] = (),
    replan: bool = False,
) -> RetrievalEvalResult:
    """categories: run only those golden groups (iteration aid; the run is
    labelled partial and never gates a merge). replan: ask the live planner
    model fresh for every item and report plans that differ from the stored
    ones (reported, never gated).

    sabotage=True runs the discrimination check on the direct search path:
    a fixed junk vector and a nonsense lexical query while the reranker
    still sees the real question — a broken retriever MUST score badly."""
    conn.execute(EVAL_SCHEMA_PATH.read_text())
    digest = corpus_hash(conn)
    run_id = conn.execute(
        "INSERT INTO eval_runs (kind, config_label, git_sha, corpus_hash) "
        "VALUES ('retrieval', %s, %s, %s) RETURNING id",
        ((config_label if not sabotage else f"{config_label}-SABOTAGE")
         + (f"-partial:{','.join(categories)}" if categories else ""), _git_sha(), digest),
    ).fetchone()[0]

    scores: list[tuple] = []  # (qid, category, metric, value, detail)
    replans: list[tuple] = []
    registry = sources.load(conn)

    items = [item for item in ablation.load_golden()
             if (not categories or item["category"] in categories) and (not items_only or item["id"] in items_only)]
    if sabotage or replan or workers <= 1:
        for item in items:
            qid, category = item["id"], item["category"]
            if sabotage:
                scores.extend((qid, category, m, v, d) for m, v, d in _score_sabotaged(conn, item, registry))
                continue
            if replan and not item.get("unanswerable"):
                stored, fresh = planner.replan(conn, item["question"], module=item.get("module"))
                if _legs_key(stored) != _legs_key(fresh.to_dict()):
                    replans.append((qid, _legs_key(stored), _legs_key(fresh.to_dict())))
            scores.extend((qid, category, m, v, d) for m, v, d in score_item(conn, item))
    else:
        for item, rows in zip(items, _score_items_parallel(items, workers), strict=True):
            scores.extend((item["id"], item["category"], m, v, d) for m, v, d in rows)

    scores.extend(_item_verdict_rows(scores))
    by_qid = {item["id"]: expected_source(item, registry) for item in ablation.load_golden()}
    scores = [(qid, cat, metric, value, {**detail, "source": by_qid.get(qid, "none")})
              for qid, cat, metric, value, detail in scores]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(run_id, q, c, m, v, json.dumps(d, default=str)) for q, c, m, v, d in scores],
        )
    conn.commit()
    result = _summarize(run_id, scores)
    result.corpus_hash = digest
    result.replan = replans
    if not sabotage:
        previous = conn.execute(
            "SELECT max(id) FROM eval_runs WHERE kind = 'retrieval' "
            "AND config_label NOT LIKE '%%SABOTAGE%%' AND id < %s", (run_id,)
        ).fetchone()[0]
        if previous is not None:
            result.diff_against = previous
            result.diff = diff_runs(conn, previous, run_id)
    return result


def _score_sabotaged(conn, item: dict, registry) -> list[tuple]:
    """The discrimination check: both retrieval arms broken on the direct
    search path (a junk vector, a nonsense lexical query), the reranker
    seeing the real question. Only items with sources take part; the
    reported hit@5 must collapse."""
    if not item.get("sources") or item.get("unanswerable"):
        return []
    junk_vector = "[" + ",".join(["0.01"] * 1536) + "]"
    junk_text = "zzqx zzqv zzqw"
    reading = planner.read_route(conn, item["question"])
    decision = router.route(item["question"], hierarchies=registry.hierarchies(), reading=reading)
    ctx = retrieval.resolve_context(conn, _member_on_screen(item), item["question"], case_id=item.get("case_id"))
    decision = retrieval.expand_versions(conn, decision, ctx, item["question"])
    decision = retrieval.bind_enrollment_plan(decision, ctx)
    question = deid.translate_query(conn, ctx.query)
    candidates = retrieval.search(conn, junk_text, junk_vector, decision, embed=lambda t: junk_vector,
                                  member_key=ctx.member_key, record=ctx.record)
    reranked = rerank.rerank(question, candidates, top_n=10, stratify_years=decision.years,
                             stratify_plans=decision.cover_keys if decision.cover_field == "plan_code" else ())
    top5 = reranked[:5]
    flags = [ablation.is_relevant(c, item["sources"]) for c in top5]
    return [("hit@5", float(any(flags)), {"sabotage": True}),
            ("precision@5", (sum(flags) / len(top5)) if top5 else 0.0, {"sabotage": True})]


# ---------------------------------------------------------------- summaries
def diff_runs(conn: psycopg.Connection, before: int, after: int) -> dict:
    """Which questions flipped, per 0/1 metric, between two eval runs. The
    number that moved is never the story; the questions that flipped are."""
    out = {}
    metrics = sorted({m for (m,) in conn.execute(
        "SELECT DISTINCT metric FROM eval_scores WHERE run_id IN (%s, %s) "
        "AND (metric = 'item_pass' OR metric LIKE 'check\\_%%' OR metric IN ('hit@5', 'source_coverage'))",
        (before, after)).fetchall()})
    for metric in metrics:
        rows = conn.execute(
            "SELECT run_id, question_id, value FROM eval_scores "
            "WHERE metric = %s AND run_id IN (%s, %s)", (metric, before, after)
        ).fetchall()
        a = {q: float(v) for r, q, v in rows if r == before}
        b = {q: float(v) for r, q, v in rows if r == after}
        if a or b:
            d = stats.paired_diff(a, b)
            if d["gained"] or d["lost"]:
                out[metric] = d
    return out


_SLICE_METRICS = ("hit@5", "precision@5", "source_coverage", "item_pass")


def _slice(rows: list[tuple]) -> dict:
    """Metrics for one slice of scores: mean, and for 0/1 metrics a 95%
    Wilson interval as `<metric>_ci`; `n` = distinct questions."""
    out = {}
    for metric in _SLICE_METRICS:
        vals = [v for _, _, m, v, _ in rows if m == metric]
        if not vals:
            continue
        out[metric] = round(sum(vals) / len(vals), 3)
        if metric in stats.BINARY_METRICS or metric == "item_pass":
            low, high = stats.wilson(int(round(sum(vals))), len(vals))
            out[f"{metric}_ci"] = (round(low, 3), round(high, 3))
    out["n"] = len({q for q, *_ in rows})
    return out


def _item_verdict_rows(scores: list[tuple]) -> list[tuple]:
    """One `item_pass` row per item (1.0 / 0.0) carrying its work category
    and group, or `item_unverified` when a check was skipped in this run
    (no warehouse). A skipped check never counts as a pass."""
    by_id = {item["id"]: item for item in ablation.load_golden()}
    rows = []
    for qid, (passed, skipped) in sorted(taxonomy.item_verdicts(scores).items()):
        item = by_id.get(qid)
        if item is None:
            continue
        detail = {"work_category": int(item["work_category"]), "group": item["category"]}
        if passed is None:
            rows.append((qid, item["category"], "item_unverified", 1.0, {**detail, "skipped": skipped}))
        else:
            rows.append((qid, item["category"], "item_pass", float(passed), detail))
    return rows


def _rate_table(scores: list[tuple], key) -> dict:
    """Pass rate per bucket from the item_pass / item_unverified rows:
    {bucket: {n, passed, unverified, rate, ci, failing}}."""
    table: dict = {}
    for qid, cat, metric, value, detail in scores:
        if metric not in ("item_pass", "item_unverified"):
            continue
        bucket = key(cat, detail)
        slot = table.setdefault(bucket, {"n": 0, "passed": 0, "unverified": 0, "failing": [], "passing": []})
        if metric == "item_unverified":
            slot["unverified"] += 1
        else:
            slot["n"] += 1
            if value >= 1.0:
                slot["passed"] += 1
                slot["passing"].append(qid)
            else:
                slot["failing"].append(qid)
    for slot in table.values():
        slot["rate"] = round(slot["passed"] / slot["n"], 3) if slot["n"] else None
        slot["ci"] = tuple(round(x, 3) for x in stats.wilson(slot["passed"], slot["n"])) if slot["n"] else None
        slot["failing"].sort()
        slot["passing"].sort()
    return table


def _summarize(run_id: int, scores: list[tuple]) -> RetrievalEvalResult:
    result = RetrievalEvalResult(run_id=run_id)

    by_work = _rate_table(scores, lambda cat, d: int(d["work_category"]))
    for number, slot in by_work.items():
        slot["name"] = taxonomy.WORK_CATEGORIES[number].name
    result.by_work = dict(sorted(by_work.items()))
    result.by_group = dict(sorted(_rate_table(scores, lambda cat, d: d.get("group", cat)).items()))

    def mean(metric, rows):
        vals = [v for _, _, m, v, _ in rows if m == metric]
        return sum(vals) / len(vals) if vals else None

    for category in sorted({c for _, c, _, _, _ in scores}):
        result.by_category[category] = _slice([s for s in scores if s[1] == category])
    for slice_name in sorted({d.get("source", "none") for *_, d in scores}):
        result.by_source[slice_name] = _slice([s for s in scores if s[4].get("source", "none") == slice_name])

    for stage in ("embed", "search", "rerank", "total"):
        vals = [v for _, _, m, v, _ in scores if m == f"latency_{stage}"]
        if vals:
            result.latency[stage] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "n": len(vals)}

    passes = [v for _, _, m, v, _ in scores if m == "item_pass"]
    result.overall = {
        "item_pass": round(sum(passes) / len(passes), 3) if passes else None,
        "hit@5": mean("hit@5", scores),
        "precision@5": mean("precision@5", scores),
        "source_coverage": mean("source_coverage", scores),
    }
    if passes:
        result.overall_ci["item_pass"] = tuple(round(x, 3) for x in stats.wilson(int(round(sum(passes))), len(passes)))
    for metric in ("hit@5",):
        vals = [v for _, _, m, v, _ in scores if m == metric]
        if vals:
            result.overall_ci[metric] = tuple(round(x, 3) for x in stats.wilson(int(round(sum(vals))), len(vals)))
    skipped = {q for q, _, m, _, _ in scores if m == WAREHOUSE_SKIPPED}
    if skipped:
        result.overall["warehouse_skipped_items"] = len(skipped)

    result.health = checks.summarize_health([d for _, _, m, _, d in scores if m == "payload_health"])
    guardrail_groups = {name for name, g in taxonomy.GROUPS.items() if g.band == "guardrail"}
    result.guardrails_passing = sorted(q for q, cat, m, v, _ in scores
                                       if m == "item_pass" and v >= 1.0 and cat in guardrail_groups)
    return result


# ---------------------------------------------------------------- the ratchet
def load_baseline(path=BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_baseline(result: RetrievalEvalResult, path=BASELINE_PATH) -> dict:
    """Store a run as the reference every later run ratchets against."""
    baseline = {
        "run_id": result.run_id,
        "git_sha": _git_sha(),
        "corpus_hash": result.corpus_hash,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": {k: v for k, v in result.overall.items() if v is not None},
        "by_work": {str(n): {"name": s["name"], "rate": s["rate"], "n": s["n"]} for n, s in result.by_work.items() if s["n"]},
        "by_group": {g: {"rate": s["rate"], "n": s["n"]} for g, s in result.by_group.items() if s["n"]},
        "guardrails_passing": result.guardrails_passing,
        # every verified item's verdict, so a run that can verify only a subset
        # (CI holds no warehouse credentials) is compared over the same items
        "items": {q: 1 for s in result.by_group.values() for q in s["passing"]}
                 | {q: 0 for s in result.by_group.values() for q in s["failing"]},
        "health": result.health,
    }
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


def ratchet(result: RetrievalEvalResult, baseline: dict | None) -> list[str]:
    """The gate: every category and group holds its baseline pass rate, and
    no guardrail item that passed at the baseline fails now. A run verifies
    only the items its credentials allow (CI has no warehouse), so each
    comparison is over the items verified in THIS run: the baseline rate is
    recomputed from the stored per-item verdicts on exactly those ids."""
    if baseline is None:
        return ["no baseline stored (eval/baseline.json): run with --write-baseline first"]
    failures = []
    verdicts = baseline.get("items") or {}

    def reference(ref: dict, now: dict) -> tuple[float | None, int]:
        """(baseline rate over the items this run verified, how many of them the baseline knew)."""
        if not verdicts:
            return ref.get("rate"), now["n"]  # an older baseline without verdicts: whole-bucket rate
        known = [q for q in now["passing"] + now["failing"] if q in verdicts]
        return (round(sum(verdicts[q] for q in known) / len(known), 3), len(known)) if known else (None, 0)

    for number, ref in baseline.get("by_work", {}).items():
        now = result.by_work.get(int(number))
        if not now or not now["n"]:
            continue
        ref_rate, known = reference(ref, now)
        now_rate = round(sum(1 for q in now["passing"] if q in verdicts) / known, 3) if verdicts and known else now["rate"]
        if ref_rate is not None and now_rate < ref_rate:
            failures.append(f"category {ref['name']}: {now_rate:.3f} < baseline {ref_rate:.3f} over the {known} items verified "
                            f"(failing: {', '.join(now['failing'][:6])})")
    for group, ref in baseline.get("by_group", {}).items():
        now = result.by_group.get(group)
        if not now or not now["n"]:
            continue
        ref_rate, known = reference(ref, now)
        now_rate = round(sum(1 for q in now["passing"] if q in verdicts) / known, 3) if verdicts and known else now["rate"]
        if ref_rate is not None and now_rate < ref_rate:
            failures.append(f"group {group}: {now_rate:.3f} < baseline {ref_rate:.3f} over the {known} items verified "
                            f"(failing: {', '.join(now['failing'][:6])})")
    verified_now = {q for s in result.by_group.values() for q in s["passing"] + s["failing"]}
    passing_now = set(result.guardrails_passing)
    regressed = [q for q in baseline.get("guardrails_passing", []) if q in verified_now and q not in passing_now]
    if regressed:
        failures.append(f"guardrail regression: {', '.join(regressed)}")
    return failures


def _legs_key(plan: dict | None) -> str:
    """Order-free summary of a plan for the replan report: shape + (kind, source/query) per leg."""
    if not plan:
        return "none"
    legs = sorted((leg["kind"], leg.get("query_name") or ",".join(sorted(leg.get("sources") or [])) or "*") for leg in plan.get("legs", []))
    return f"{plan.get('shape')}: " + "; ".join(f"{k}[{t}]" for k, t in legs)
