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
    categories: tuple[str, ...] = (),
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
    registry = sources.load(conn)
    junk_vector = "[" + ",".join(["0.01"] * 1536) + "]"
    junk_text = "zzqx zzqv zzqw"  # matches no chunk: the lexical arm returns nothing

    for item in ablation.load_golden():
        qid, category = item["id"], item["category"]
        if categories and category not in categories:
            continue

        if category == "two_lane":
            continue  # needs Snowflake; asserted in tests/test_two_lane_golden.py

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
            payload = run_query(conn, item["question"], persona="employee", source="eval",
                                member_id=item.get("member_id"))
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

            denied = run_query(conn, item["question"], persona=item["persona_deny"],
                               source="eval", member_id=item.get("member_id"))
            allowed = run_query(conn, item["question"], persona=item["persona_allow"],
                                source="eval", member_id=item.get("member_id"))
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
        member_key = retrieval.resolve_member(conn, item.get("member_id"), item["question"])
        # The eval runs as admin, which is entitled to the vault: translate
        # like the pipeline does, so tokenized notes stay reachable by the
        # identifiers a question naturally uses.
        question = deid.translate_query(conn, item["question"])

        if item.get("unanswerable"):
            expected_gate = item["expected_trigger"] == "scope_gate"
            gated = decision.scope != "in_scope"
            scores.append((qid, category, "gate_correct", float(gated == expected_gate), {}))
            if not gated:
                vector = junk_vector if sabotage else retrieval.embed_cached(conn, question)
                candidates = retrieval.search(conn, junk_text if sabotage else question, vector, decision,
                                              embed=(lambda t: junk_vector) if sabotage else (lambda t: retrieval.embed_cached(conn, t)),
                                              member_key=member_key)
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
                                              member_key=member_key)
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
                  "deny_clean", "allow_answered", "allow_hit", "scope_clean")


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
    scope = mean("scope_clean", scores)
    if scope is not None:
        result.overall["scope_clean"] = scope
        if scope < THRESHOLDS["scope_clean"]:
            result.failures.append("member scope leaked: another member's record was returned")
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
