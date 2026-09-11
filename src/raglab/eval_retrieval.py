"""Tier-1 evaluation: deterministic retrieval metrics over the golden set.

Free to run, reproducible, writes to the metrics tables, and (with gate
thresholds) fails loudly on regression — this is what CI enforces and what
every A/B experiment is decided on.

Metrics per answerable question:
- hit@5            — any relevant chunk in the post-rerank top 5
- precision@5      — fraction of the post-rerank top 5 that is relevant
- source_coverage  — fraction of expected sources represented in top 10
                     (YoY questions need BOTH years; this catches one-year
                     shortcuts)
- abstained        — whether the funnel wrongly abstained (must be 0)

Per unanswerable question:
- gate_correct     — scope gate fired iff expected
- abstained        — for low-confidence-type questions, funnel must abstain

Per persona_negative question (entitlement assertions, run through the full
persona pipeline — RLS, vault translation, disclosure — in both directions):
- deny_clean       — the protected document is absent from the unauthorized
                     persona's results (it may abstain, or answer from what
                     it is entitled to)
- allow_answered   — the authorized persona must get status ok
- allow_hit        — the authorized answer cites the expected document
"""

import hashlib
import subprocess
from dataclasses import dataclass, field

import psycopg

from raglab import planner
from raglab import ablation, config, deid, rerank, retrieval, router, sources, stats
from raglab.timing import BUDGET_P95_MS, Stopwatch, percentile

EVAL_SCHEMA_PATH = config.REPO_ROOT / "db" / "eval.sql"

# Gate thresholds — a regression below any of these fails the build.
# Entitlement metrics are absolute: a single persona leak or blocked
# authorized answer fails the run.
THRESHOLDS = {
    "hit@5": 0.85,
    "gate_correct": 1.0,
    "wrong_abstention_rate": 0.05,
    "deny_clean": 1.0,
    "allow_answered": 1.0,
    # Ratchet: a floor raised whenever a fix lands, so year-over-year
    # coverage cannot slip unnoticed. History: 0.688 in v1 (the golden set
    # listed one page per year; the reranker preferred "changes" sections);
    # 2026-09-09: each year's label accepts every page that states that
    # year's value, and prior years are searched with a year-neutral form of
    # the question (router.year_queries) → measured 1.0 on 8 questions. The
    # floor sits one half-miss below (7.5/8) so a single borderline page is
    # a finding, not a red build.
    "yoy_source_coverage": 0.9,
    # Member scoping: a question about one member never returns another
    # member's records. Absolute.
    "scope_clean": 1.0,
    # Version precedence: no superseded policy version in a default top-10.
    "version_clean": 1.0,
    # Phase 3 (decision 6): the planner's legs equal the item's expected legs
    # (type + source/query, order-free) — plans are stored, so a miss is a
    # defect in menu, prompt, or item, not variance. Absolute.
    "routing_accuracy": 1.0,
    # Every required leg returned its evidence (needs the warehouse: skipped
    # where no credentials, reported as such).
    "complete_recall": 0.9,
    # Adversarial cases (D15): partial entitlement never answers from half;
    # contradictory sources both surface.
    "adversarial_ok": 1.0,
}


@dataclass
class RetrievalEvalResult:
    run_id: int
    by_category: dict = field(default_factory=dict)
    by_source: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)
    overall_ci: dict = field(default_factory=dict)  # metric -> (low, high), 95% Wilson
    diff: dict = field(default_factory=dict)        # vs the previous run: metric -> paired_diff
    latency: dict = field(default_factory=dict)     # stage -> {p50, p95} ms over the golden set
    diff_against: int | None = None
    failures: list = field(default_factory=list)
    replan: list = field(default_factory=list)  # [(qid, stored shape/legs, fresh shape/legs)] — plans the live model would change
    corpus_hash: str = ""


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
            # a path under a source directory — data/internal/ (authored,
            # generated) or data/raw/ (fetched: carrier letters)
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


def run(
    conn: psycopg.Connection,
    config_label: str = "baseline",
    sabotage: bool = False,
    categories: tuple[str, ...] = (),
    replan: bool = False,
) -> RetrievalEvalResult:
    """categories: run only those golden categories (iteration aid; the
    run is labelled partial and never gates a merge).

    sabotage=True breaks BOTH retrieval arms — a fixed junk vector and a
    nonsense lexical query — while the reranker still sees the real
    question. The discrimination check: a broken retriever MUST score badly.
    (Vector-only sabotage stopped discriminating once BM25 landed: the
    lexical arm alone finds 24/29 and the reranker sorts them.)"""
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
    junk_vector = "[" + ",".join(["0.01"] * 1536) + "]"
    junk_text = "zzqx zzqv zzqw"  # matches no chunk: the lexical arm returns nothing

    for item in ablation.load_golden():
        qid, category = item["id"], item["category"]
        if categories and category not in categories:
            continue
        if category == "named_query":  # warehouse-only assertions live in tests/test_named_queries.py
            continue

        # Planner shape (Phase 3): every golden item has an expected shape —
        # compound for the two-lane items, simple for everything else. The
        # planner's decision is read from the stored plan (the model ran once
        # per question) or from the rules path by configuration. Reported now;
        # routing accuracy against expected legs gates from P3-PR4.
        if category not in ("unanswerable",):
            expected_shape = item.get("expected_shape") or ("compound" if category == "compound" else "simple")
            plan = planner.plan_for(conn, item["question"], module=item.get("module"))
            if replan:
                stored, fresh = planner.replan(conn, item["question"])
                if _legs_key(stored) != _legs_key(fresh.to_dict()):
                    replans.append((qid, _legs_key(stored), _legs_key(fresh.to_dict())))
            scores.append((qid, category, "shape_accuracy", float(plan.shape == expected_shape),
                           {"shape": plan.shape, "origin": plan.origin, "legs": len(plan.legs),
                            "fallback": plan.fallback_reason}))

        if category in ("compound", "adversarial"):
            scores.extend(_score_composed(conn, item))
            continue

        if category == "version_negative":
            # A default (undated) question about a policy must not surface
            # its superseded version; a dated one must not surface the other.
            decision = router.route(item["question"])
            ctx = retrieval.resolve_context(conn, None, item["question"])
            decision = retrieval.expand_versions(conn, decision, ctx, item["question"])
            question = deid.translate_query(conn, ctx.query)
            vector = junk_vector if sabotage else retrieval.embed_cached(conn, question)
            candidates = retrieval.search(conn, junk_text if sabotage else question, vector, decision,
                                          embed=(lambda t: junk_vector) if sabotage else (lambda t: retrieval.embed_cached(conn, t)),
                                          member_key=ctx.member_key, record=ctx.record)
            reranked = rerank.rerank(question, candidates, top_n=10, stratify_years=decision.years)
            titles = [c.doc_title for c in reranked[:10]]
            leaked = [t for t in titles if any(a in t for a in item["absent_titles"])]
            scores.append((qid, category, "version_clean", float(not leaked), {"leaked": leaked, "as_of": decision.as_of}))
            continue

        if category == "scope_negative":
            # Member scoping: every chunk returned for a question about
            # member X belongs to X or to a source that is not member-scoped.
            # Runs the real employee path (identifier translation included).
            if sabotage:
                continue
            from raglab.pipeline import run_query

            person = conn.execute(
                "SELECT id FROM synthea.patients WHERE member_id = %s", (item["member_id"],)
            ).fetchone()
            # The governance gates test the WALLS under the composed path (one
            # payload, one disclosure, spec 1.1.0), not routing: the plan is the
            # rules plan — one document probe on the question. Routing quality is
            # its own metric (shape_accuracy now; routing accuracy in P3-PR4).
            payload = planner.compose(conn, item["question"], planner.Caller(persona=item.get("persona", "member_services")),
                                      member_id=item.get("member_id"), plan=planner.plan_rules(item["question"]), source="eval")
            hashes = [c["source"]["content_hash"] for c in payload.get("chunks", [])]
            others = 0
            if hashes:
                others = conn.execute(
                    "SELECT count(*) FROM documents WHERE content_hash = ANY(%s) "
                    "AND member_key IS NOT NULL AND member_key::text <> %s",
                    (hashes, person[0] if person else ""),
                ).fetchone()[0]
            scores.append((qid, category, "scope_clean", float(others == 0),
                           {"foreign_chunks": others, "returned": len(hashes)}))
            continue

        if category == "persona_negative":
            # Entitlement assertions exercise the REAL persona path
            # (SET ROLE, vault translation, disclosure log) — a junk vector
            # can't stand in for it, so sabotage runs skip them.
            if sabotage:
                continue
            from raglab.pipeline import run_query

            denied = planner.compose(conn, item["question"], planner.Caller(persona=item["persona_deny"]),
                                     member_id=item.get("member_id"), plan=planner.plan_rules(item["question"]), source="eval")
            allowed = planner.compose(conn, item["question"], planner.Caller(persona=item["persona_allow"]),
                                      member_id=item.get("member_id"), plan=planner.plan_rules(item["question"]), source="eval")
            titles = [c["source"]["title"] for c in allowed.get("chunks", [])]
            denied_titles = [c["source"]["title"] for c in denied.get("chunks", [])]
            leaked = [t for t in denied_titles if any(e in t for e in item["allow_titles"])]
            allow_hit = float(any(
                expected in title
                for expected in item["allow_titles"] for title in titles[:5]
            ))
            scores.append((qid, category, "deny_clean", float(not leaked),
                           {"persona": item["persona_deny"], "status": denied["status"],
                            "leaked": leaked}))
            scores.append((qid, category, "allow_answered",
                           float(allowed["status"] == "ok"),
                           {"persona": item["persona_allow"],
                            "confidence": allowed.get("confidence")}))
            scores.append((qid, category, "allow_hit", allow_hit,
                           {"expected": item["allow_titles"]}))
            continue

        decision = router.route(item["question"])
        # Member context, like the pipeline: the item's member_id field (the
        # member a rep would have open) or an identifier in the question.
        ctx = retrieval.resolve_context(conn, item.get("member_id"), item["question"])
        decision = retrieval.expand_versions(conn, decision, ctx, item["question"])
        member_key, record = ctx.member_key, ctx.record
        # The eval runs as admin, which is entitled to the vault: translate
        # like the pipeline does (names -> pseudonyms); resolved identifiers
        # are already context, not search words.
        question = deid.translate_query(conn, ctx.query)

        if item.get("unanswerable"):
            expected_gate = item["expected_trigger"] == "scope_gate"
            gated = decision.scope != "in_scope"
            scores.append((qid, category, "gate_correct", float(gated == expected_gate), {}))
            if not gated:
                vector = junk_vector if sabotage else retrieval.embed_cached(conn, question)
                candidates = retrieval.search(conn, junk_text if sabotage else question, vector, decision,
                                              embed=(lambda t: junk_vector) if sabotage else (lambda t: retrieval.embed_cached(conn, t)),
                                              member_key=member_key, record=record)
                reranked = rerank.rerank(question, candidates)
                abstained, best = rerank.abstention_verdict(reranked)
                scores.append((qid, category, "abstained", float(abstained),
                               {"best_score": round(best, 4)}))
            continue

        watch = Stopwatch()
        with watch.stage("embed"):
            vector = junk_vector if sabotage else retrieval.embed_cached(conn, question)
        with watch.stage("search"):
            candidates = retrieval.search(conn, junk_text if sabotage else question, vector, decision,
                                              embed=(lambda t: junk_vector) if sabotage else (lambda t: retrieval.embed_cached(conn, t)),
                                              member_key=member_key, record=record)
        with watch.stage("rerank"):
            reranked = rerank.rerank(
                question, candidates, top_n=10,
                stratify_years=decision.years,
            )
        abstained, best = rerank.abstention_verdict(reranked)
        snap = watch.snapshot()
        for stage in ("embed", "search", "rerank", "total"):
            scores.append((qid, category, f"latency_{stage}", snap[stage], {"host": snap["host"]}))

        top5 = reranked[:5]
        relevant_flags = [ablation.is_relevant(c, item["sources"]) for c in top5]
        hit = float(any(relevant_flags))
        precision = sum(relevant_flags) / len(top5) if top5 else 0.0
        covered = sum(
            1 for source in item["sources"]
            if any(ablation.is_relevant(c, [source]) for c in reranked[:10])
        )
        coverage = covered / len(item["sources"])

        scores.append((qid, category, "hit@5", hit, {}))
        scores.append((qid, category, "precision@5", precision, {}))
        scores.append((qid, category, "source_coverage", coverage,
                       {"expected_sources": len(item["sources"])}))
        scores.append((qid, category, "wrong_abstention", float(abstained),
                       {"best_score": round(best, 4)}))

    import json as _json

    by_qid = {item["id"]: expected_source(item, registry) for item in ablation.load_golden()}
    scores = [
        (qid, cat, metric, value, {**detail, "source": by_qid.get(qid, "none")})
        for qid, cat, metric, value, detail in scores
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(run_id, q, c, m, v, _json.dumps(d)) for q, c, m, v, d in scores],
        )
    conn.commit()
    result = _summarize(run_id, scores)
    result.corpus_hash = digest
    if not sabotage:
        previous = conn.execute(
            "SELECT max(id) FROM eval_runs WHERE kind = 'retrieval' "
            "AND config_label NOT LIKE '%%SABOTAGE%%' AND id < %s", (run_id,)
        ).fetchone()[0]
        if previous is not None:
            result.diff_against = previous
            result.diff = diff_runs(conn, previous, run_id)
    return result


def diff_runs(conn: psycopg.Connection, before: int, after: int) -> dict:
    """Which questions flipped, per 0/1 metric, between two eval runs. The
    number that moved is never the story; the questions that flipped are."""
    out = {}
    for metric in sorted(stats.BINARY_METRICS):
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


_SLICE_METRICS = ("hit@5", "precision@5", "source_coverage", "gate_correct",
                  "deny_clean", "allow_answered", "allow_hit", "scope_clean", "version_clean", "shape_accuracy",
                  "routing_accuracy", "complete_recall", "widened_rescue", "adversarial_ok")


def _slice(rows: list[tuple]) -> dict:
    """Metrics for one slice of scores: mean, and for 0/1 metrics a 95%
    Wilson interval as `<metric>_ci`; `n` = distinct questions."""
    out = {}
    for metric in _SLICE_METRICS:
        vals = [v for _, _, m, v, _ in rows if m == metric]
        if not vals:
            continue
        out[metric] = round(sum(vals) / len(vals), 3)
        if metric in stats.BINARY_METRICS:
            low, high = stats.wilson(int(round(sum(vals))), len(vals))
            out[f"{metric}_ci"] = (round(low, 3), round(high, 3))
    out["n"] = len({q for q, *_ in rows})
    return out


def _summarize(run_id: int, scores: list[tuple]) -> RetrievalEvalResult:
    result = RetrievalEvalResult(run_id=run_id)

    def mean(metric, rows):
        vals = [v for _, _, m, v, _ in rows if m == metric]
        return sum(vals) / len(vals) if vals else None

    categories = sorted({c for _, c, _, _, _ in scores})
    for category in categories:
        result.by_category[category] = _slice([s for s in scores if s[1] == category])

    # Per-source slice: which source the expected evidence lives in. Every
    # retrieval metric, reported by source, so a new source cannot degrade an
    # old one without a number moving.
    slices = sorted({d.get("source", "none") for *_, d in scores})
    for slice_name in slices:
        result.by_source[slice_name] = _slice(
            [s for s in scores if s[4].get("source", "none") == slice_name]
        )

    for stage in ("embed", "search", "rerank", "total"):
        vals = [v for _, _, m, v, _ in scores if m == f"latency_{stage}"]
        if vals:
            result.latency[stage] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "n": len(vals)}

    result.overall = {
        "hit@5": mean("hit@5", scores),
        "precision@5": mean("precision@5", scores),
        "source_coverage": mean("source_coverage", scores),
        "gate_correct": mean("gate_correct", scores),
        "wrong_abstention_rate": mean("wrong_abstention", scores),
    }
    for metric in ("hit@5", "gate_correct"):
        vals = [v for _, _, m, v, _ in scores if m == metric]
        if vals:
            result.overall_ci[metric] = tuple(
                round(x, 3) for x in stats.wilson(int(round(sum(vals))), len(vals))
            )

    if result.overall["hit@5"] is not None and result.overall["hit@5"] < THRESHOLDS["hit@5"]:
        result.failures.append(
            f"hit@5 {result.overall['hit@5']:.3f} < {THRESHOLDS['hit@5']}"
        )
    if result.overall["gate_correct"] is not None and result.overall["gate_correct"] < THRESHOLDS["gate_correct"]:
        result.failures.append("scope gate leaked")
    if (result.overall["wrong_abstention_rate"] or 0) > THRESHOLDS["wrong_abstention_rate"]:
        result.failures.append(
            f"wrong abstentions {result.overall['wrong_abstention_rate']:.3f}"
        )
    yoy = result.by_category.get("yoy", {}).get("source_coverage")
    if yoy is not None and yoy < THRESHOLDS["yoy_source_coverage"]:
        result.failures.append(
            f"yoy source_coverage {yoy:.3f} < ratchet {THRESHOLDS['yoy_source_coverage']}"
        )
    version = mean("version_clean", scores)
    if version is not None:
        result.overall["version_clean"] = version
        if version < THRESHOLDS["version_clean"]:
            result.failures.append("version leak: a superseded policy version reached a default top-10")
    scope = mean("scope_clean", scores)
    if scope is not None:
        result.overall["scope_clean"] = scope
        if scope < THRESHOLDS["scope_clean"]:
            result.failures.append("member scope leaked: another member's record was returned")
    # Phase 3 gates (decision 6): routing 1.0, complete recall 0.9, adversarial 1.0.
    for metric, message in (("routing_accuracy", "planner routed a compound question to the wrong legs"),
                            ("complete_recall", "a required leg did not return its evidence"),
                            ("adversarial_ok", "adversarial case failed (answered from half, or a conflicting source hidden)")):
        value = mean(metric, scores)
        if value is not None:
            result.overall[metric] = round(value, 3)
            if value < THRESHOLDS[metric]:
                result.failures.append(f"{message}: {metric} {value:.3f} < {THRESHOLDS[metric]}")
    skipped = [q for q, _, m, _, _ in scores if m == "complete_recall_skipped"]
    if skipped:
        result.overall["complete_recall_skipped"] = len(skipped)
    for metric in ("deny_clean", "allow_answered"):
        value = mean(metric, scores)
        result.overall[metric] = value
        if value is not None and value < THRESHOLDS[metric]:
            result.failures.append(
                "persona leak: an unauthorized persona received content"
                if metric == "deny_clean"
                else "entitled persona was wrongly blocked"
            )
    return result


def _legs_key(plan: dict | None) -> str:
    """Order-free summary of a plan for the replan report: shape + (kind, source/query) per leg."""
    if not plan:
        return "none"
    legs = sorted((leg["kind"], leg.get("query_name") or ",".join(sorted(leg.get("sources") or [])) or "*") for leg in plan.get("legs", []))
    return f"{plan.get('shape')}: " + "; ".join(f"{k}[{t}]" for k, t in legs)


def _legs_multiset(legs: list[dict]) -> list[tuple]:
    """(kind, source-or-query) per leg, order-free. A document leg with no
    hints matches an expected leg on any source ('*')."""
    out = []
    for leg in legs:
        if leg["kind"] == "member_query":
            out.append(("member_query", leg.get("query_name")))
        else:
            out.append(("doc_probe", ",".join(sorted(leg.get("sources") or [])) or "*"))
    return sorted(out)


def _leg_matches(planned: dict, expected: dict) -> bool:
    if planned["kind"] != expected["kind"]:
        return False
    if expected["kind"] == "member_query":
        allowed = set(expected.get("query_name_any") or [expected.get("query_name")])
        return planned.get("query_name") in allowed
    got = set(planned.get("sources") or [])
    want = set(expected.get("sources") or []) | set(expected.get("sources_any") or [])
    return not got or not want or bool(got & want)  # a hint-less leg matches any expected source


def _routing_matches(planned: list[dict], expected: list[dict], optional: list[dict] = ()) -> bool:
    """Order-free: every expected leg is matched by a distinct planned leg
    and nothing is left over. An expected member leg may name alternatives
    (`query_name_any`) when more than one catalog query answers the same
    part of the question."""
    remaining = list(planned)
    for exp in expected:
        match = next((p for p in remaining if _leg_matches(p, exp)), None)
        if match is None:
            return False
        remaining.remove(match)
    # legs the item allows but does not require (a supporting warehouse row, a document beside the record)
    for opt in optional:
        match = next((p for p in remaining if _leg_matches(p, opt)), None)
        if match is not None:
            remaining.remove(match)
    return not remaining


def _score_composed(conn, item: dict) -> list[tuple]:
    """Compound + adversarial items (Phase 3): the planner's ROUTING is scored
    from the plan alone (no warehouse needed); EXECUTION — complete recall,
    widened rescue, adversarial status — runs the composed path with the
    model's plan as the identity's persona + warehouse role, and is skipped
    (reported) where the warehouse is not reachable."""
    import os

    from raglab.mcp_server import IDENTITIES

    qid, category = item["id"], item["category"]
    persona, role = IDENTITIES[item["identity"]]
    detail = {"source": "none", "identity": item["identity"]}
    scores: list[tuple] = []
    module = item.get("module")
    detail["module"] = module
    plan = planner.plan_for(conn, item["question"], module=module)
    plan_dict = plan.to_dict()
    if "expected_legs" in item:
        scores.append((qid, category, "routing_accuracy",
                       float(_routing_matches(plan_dict["legs"], item["expected_legs"], item.get("optional_legs", []))),
                       {**detail, "planned": _legs_multiset(plan_dict["legs"]), "expected": _legs_multiset(item["expected_legs"]),
                        "origin": plan.origin, "fallback": plan.fallback_reason}))
    warehouse_needed = any(l["kind"] == "member_query" for l in plan_dict["legs"]) and role is not None
    if warehouse_needed and not os.environ.get("SNOWFLAKE_ACCOUNT"):
        scores.append((qid, category, "complete_recall_skipped", 1.0, {**detail, "reason": "no warehouse credentials"}))
        return scores
    payload = planner.compose(conn, item["question"], planner.Caller(persona=persona, warehouse_role=role),
                              member_id=item.get("member_id"), source="eval", module=module)
    titles = [c["source"]["title"] for c in payload.get("chunks", [])]
    doc_types = {c["source"].get("doc_type") for c in payload.get("chunks", [])}
    if category == "compound":
        need = item.get("required_evidence", {})
        anchors_ok = all(any(a in t for t in titles) for a in need.get("doc_anchors", []))
        sources_ok = all(src in doc_types for src in need.get("doc_sources", []))
        # rows_min keys may be "a|b": any listed query satisfying the minimum counts
        rows_ok = all(any(w.get("query_name") in q.split("|") and (w.get("row_count") or 0) >= n for w in payload.get("warehouse_results", []))
                      for q, n in need.get("rows_min", {}).items())
        scores.append((qid, category, "complete_recall", float(payload["status"] == "ok" and anchors_ok and sources_ok and rows_ok),
                       {**detail, "status": payload["status"], "missing": payload.get("missing"), "anchors_ok": anchors_ok,
                        "sources_ok": sources_ok, "rows_ok": rows_ok}))
        scores.append((qid, category, "widened_rescue", float(payload["plan"].get("widened", False)), detail))
    else:
        ok = True
        if "expect_status_any" in item:
            ok &= payload["status"] in item["expect_status_any"]
        elif "expect_status" in item:
            ok &= payload["status"] == item["expect_status"]
        if "expect_never_source" in item:
            ok &= item["expect_never_source"] not in doc_types
        if "expect_sources_present" in item:
            ok &= all(src in doc_types for src in item["expect_sources_present"])
        if "expect_never_query" in item:
            ok &= all(w.get("query_name") != item["expect_never_query"] for w in payload.get("warehouse_results", []))
            ok &= all(l.get("query_name") != item["expect_never_query"] for l in plan_dict["legs"])
        if "expect_missing_query" in item:
            ok &= any(w.get("query_name") == item["expect_missing_query"] and w.get("status") != "ok"
                      for w in payload.get("warehouse_results", []))
        scores.append((qid, category, "adversarial_ok", float(ok),
                       {**detail, "kind": item.get("kind"), "status": payload["status"], "doc_types": sorted(t for t in doc_types if t),
                        "missing": payload.get("missing")}))
    return scores
