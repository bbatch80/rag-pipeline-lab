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
- deny_abstained   — the unauthorized persona must get insufficient_evidence
- allow_answered   — the authorized persona must get status ok
- allow_hit        — the authorized answer cites the expected document
"""

import hashlib
import subprocess
from dataclasses import dataclass, field

import psycopg

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
    "deny_abstained": 1.0,
    "allow_answered": 1.0,
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
            rel = "data/internal/" + source["internal"]
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
) -> RetrievalEvalResult:
    """sabotage=True replaces query vectors with a fixed junk vector — the
    discrimination check: a broken retriever MUST score badly."""
    conn.execute(EVAL_SCHEMA_PATH.read_text())
    digest = corpus_hash(conn)
    run_id = conn.execute(
        "INSERT INTO eval_runs (kind, config_label, git_sha, corpus_hash) "
        "VALUES ('retrieval', %s, %s, %s) RETURNING id",
        (config_label if not sabotage else f"{config_label}-SABOTAGE", _git_sha(), digest),
    ).fetchone()[0]

    scores: list[tuple] = []  # (qid, category, metric, value, detail)
    registry = sources.load(conn)
    junk_vector = "[" + ",".join(["0.01"] * 1536) + "]"

    for item in ablation.load_golden():
        qid, category = item["id"], item["category"]

        if category == "two_lane":
            continue  # needs Snowflake; asserted in tests/test_two_lane_golden.py

        if category == "persona_negative":
            # Entitlement assertions exercise the REAL persona path
            # (SET ROLE, vault translation, disclosure log) — a junk vector
            # can't stand in for it, so sabotage runs skip them.
            if sabotage:
                continue
            from raglab.pipeline import run_query

            denied = run_query(conn, item["question"],
                               persona=item["persona_deny"], source="eval")
            allowed = run_query(conn, item["question"],
                                persona=item["persona_allow"], source="eval")
            titles = [c["source"]["title"] for c in allowed.get("chunks", [])]
            allow_hit = float(any(
                expected in title
                for expected in item["allow_titles"] for title in titles[:5]
            ))
            scores.append((qid, category, "deny_abstained",
                           float(denied["status"] == "insufficient_evidence"),
                           {"persona": item["persona_deny"],
                            "confidence": denied.get("confidence")}))
            scores.append((qid, category, "allow_answered",
                           float(allowed["status"] == "ok"),
                           {"persona": item["persona_allow"],
                            "confidence": allowed.get("confidence")}))
            scores.append((qid, category, "allow_hit", allow_hit,
                           {"expected": item["allow_titles"]}))
            continue

        decision = router.route(item["question"])
        # The eval runs as admin, which is entitled to the vault: translate
        # like the pipeline does, so tokenized notes stay reachable by the
        # identifiers a question naturally uses.
        question = deid.translate_query(conn, item["question"])

        if item.get("unanswerable"):
            expected_gate = item["expected_trigger"] == "scope_gate"
            gated = decision.scope != "in_scope"
            scores.append((qid, category, "gate_correct", float(gated == expected_gate), {}))
            if not gated:
                vector = junk_vector if sabotage else retrieval.embed_query(question)
                candidates = retrieval.search(conn, question, vector, decision)
                reranked = rerank.rerank(question, candidates)
                abstained, best = rerank.abstention_verdict(reranked)
                scores.append((qid, category, "abstained", float(abstained),
                               {"best_score": round(best, 4)}))
            continue

        watch = Stopwatch()
        with watch.stage("embed"):
            vector = junk_vector if sabotage else retrieval.embed_query(question)
        with watch.stage("search"):
            candidates = retrieval.search(conn, question, vector, decision)
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
                  "deny_abstained", "allow_answered", "allow_hit")


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
    for metric in ("deny_abstained", "allow_answered"):
        value = mean(metric, scores)
        result.overall[metric] = value
        if value is not None and value < THRESHOLDS[metric]:
            result.failures.append(
                "persona leak: an unauthorized persona received content"
                if metric == "deny_abstained"
                else "entitled persona was wrongly blocked"
            )
    return result
