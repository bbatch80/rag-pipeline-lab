"""Reranker bake-off v2 (Phase 3.5, 2026-09-12): candidate rerankers, each
run through the SAME gate on the golden set and the rebuilt items, with its
abstention thresholds re-derived from its own scores, plus two ranking
metrics the gate never asked: the tie rate at the top and the margin
between first and second. Axis 2: the same models scored on re-identified
text. Nothing touches the corpus; a reranker is a scoring choice.

Targets on record: factual-05 (paraphrase), call_note-04 (paraphrase in
the record lane, 0.0002 vs 0.989 for the note's own words), call_note-03
(0.12, a hair over the bar), clinical_policy-02 (CGM: right policy first
at 0.12 under the 0.5 bar), C12, F1 (ties at 1.000), F6 (a table of
contents outscoring the answer).
"""

# The named targets: (item id, file) — the report shows where each target's
# answer chunk lands and what it scores under every candidate, not just
# the pass/fail verdict.
TARGETS = (
    ("factual-05", "rebuild"), ("call_note-04", "rebuild"), ("call_note-03", "rebuild"),
    ("clinical_policy-02", "rebuild"), ("clinical_policy-03", "rebuild"), ("table-04", "rebuild"),
    ("internal_table-02", "rebuild"), ("unanswerable-02", "rebuild"), ("unanswerable-03", "rebuild"), ("unanswerable-05", "rebuild"), ("persona_negative-02", "rebuild"), ("persona_negative-05", "rebuild"), ("persona_negative-06", "rebuild"),
    ("C12", "golden"), ("F1", "golden"), ("F6", "golden"),
)

import json
import os
import statistics
import subprocess
import sys

import psycopg

from raglab import config, rerank

REBUILD_PATH = config.REPO_ROOT / "eval" / "golden_rebuild.jsonl"
TIE_EPS = 0.005


def prepare(keys: tuple[str, ...], log=print) -> dict:
    """Download each candidate's weights (explicit, one-time) and prove it
    loads and scores; report per candidate so a failure is visible."""
    from huggingface_hub import snapshot_download

    out = {}
    for key in keys:
        spec = rerank.RERANKERS[key]
        try:
            snapshot_download(spec["model"])
            model = rerank.load_reranker(key)
            s = model.predict([("Do I need a referral to see a specialist?", "No referral needed to see a specialist.")])
            out[key] = f"ok (probe score {float(s[0]):.3f})"
        except Exception as exc:  # noqa: BLE001
            out[key] = f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"
        log(f"{key}: {out[key]}")
    return out


def run_gate(key: str, reidentify: bool, label: str) -> int | None:
    """One full eval under the candidate (a subprocess: the reranker module
    binds its model at import). Returns the run id from the receipt."""
    env = {**os.environ, "RAGLAB_RERANKER": key, "RAGLAB_RERANK_REIDENTIFY": "on" if reidentify else "off"}
    proc = subprocess.run([sys.executable, "-m", "raglab.cli", "eval-retrieval", "--label", label],
                          env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.strip().startswith("run id"):
            return int(line.split()[-1])
    return None


def derive_thresholds(conn: psycopg.Connection, run_id: int) -> dict:
    """Per model: the midpoint between the lowest best-score of an answered
    prose item and the highest best-score of an unanswerable item (the
    Phase 0 method); the record bar from the record-lane items' lowest
    answered best-score with a margin. All from the run's stored scores."""
    rows = conn.execute(
        "SELECT question_id, category, metric, value, detail FROM eval_scores WHERE run_id = %s "
        "AND metric IN ('wrong_abstention', 'abstained', 'hit@5')", (run_id,)).fetchall()
    best = {}
    for qid, cat, metric, value, detail in rows:
        if metric in ("wrong_abstention", "abstained") and "best_score" in (detail or {}):
            best[qid] = (cat, float(detail["best_score"]), metric == "abstained")
    prose_ok = [b for q, (c, b, trap) in best.items() if not trap and c not in ("call_note", "appeal", "clinical_note")]
    record_ok = [b for q, (c, b, trap) in best.items() if not trap and c in ("call_note", "appeal", "clinical_note")]
    traps = [b for q, (c, b, trap) in best.items() if trap]
    prose = round((min(prose_ok) + max(traps)) / 2, 3) if prose_ok and traps else None
    record = round(min(record_ok) * 0.8, 3) if record_ok else None
    return {"prose": prose, "record": record, "answered_prose_min": round(min(prose_ok), 3) if prose_ok else None,
            "trap_max": round(max(traps), 3) if traps else None, "answered_record_min": round(min(record_ok), 3) if record_ok else None,
            "separation": round(min(prose_ok) - max(traps), 3) if prose_ok and traps else None}


def ranking_metrics(conn: psycopg.Connection, run_id: int) -> dict:
    """Tie rate: share of answered items whose top two scores are within
    TIE_EPS. Margin: median (top1 - top2)."""
    rows = conn.execute("SELECT detail FROM eval_scores WHERE run_id = %s AND metric = 'hit@5'", (run_id,)).fetchall()
    gaps = []
    for (detail,) in rows:
        ts = (detail or {}).get("top_scores") or []
        if len(ts) >= 2:
            gaps.append(ts[0] - ts[1])
    if not gaps:
        return {"tie_rate": None, "median_margin": None, "n": 0}
    return {"tie_rate": round(sum(1 for g in gaps if g < TIE_EPS) / len(gaps), 3),
            "median_margin": round(statistics.median(gaps), 3), "n": len(gaps)}


def gate_summary(conn: psycopg.Connection, run_id: int) -> dict:
    rows = conn.execute("SELECT metric, round(avg(value), 3) FROM eval_scores WHERE run_id = %s GROUP BY 1", (run_id,)).fetchall()
    d = {m: float(v) for m, v in rows}
    return {k: d.get(k) for k in ("hit@5", "precision@5", "source_coverage", "routing_accuracy", "complete_recall",
                                   "deny_clean", "allow_answered", "scope_clean", "version_clean", "gate_correct", "wrong_abstention")}


def rebuild_check(conn: psycopg.Connection, key: str, reidentify: bool) -> dict:
    """The rebuilt golden items through the live pipeline under the
    candidate: page in the top five, required terms with it, status ok,
    and for member items no other member's record anywhere."""
    env = {**os.environ, "RAGLAB_RERANKER": key, "RAGLAB_RERANK_REIDENTIFY": "on" if reidentify else "off"}
    proc = subprocess.run([sys.executable, "-c", _REBUILD_RUNNER], env=env, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": proc.stderr[-400:]}


_REBUILD_RUNNER = r'''
import json, re
from raglab import db, planner
from raglab.ablation import PRINTED_TO_PDF_OFFSET, PAGE_TOLERANCE
items = [json.loads(l) for l in open("eval/golden_rebuild.jsonl") if l.strip()]
c = db.connect(); out = {}
def norm(t): return re.sub(r"\s+", " ", t).lower()
for it in items:
    built = planner.compose(c, it["question"], planner.Caller(persona=it.get("persona", "public")),
                            member_id=it.get("member_id"), module=it.get("module", "ask"), source="eval")
    chunks = built.get("chunks", []); top5 = chunks[:5]
    ok = built["status"] == "ok"; hit = False; leak = False
    for ch in top5:
        src = ch["source"]; text = ch.get("text") or ch.get("content") or ""
        for exp in it["sources"]:
            if "internal" in exp:
                if str(src.get("path", src.get("source_path", ""))).endswith(exp["internal"]) or exp["internal"].split("/")[-1].replace(".md", "") in src.get("title", ""):
                    hit = True
            else:
                pages = set(src.get("pages") or [])
                wanted = {p + PRINTED_TO_PDF_OFFSET + d for p in exp["printed_pages"] for d in range(-PAGE_TOLERANCE, PAGE_TOLERANCE + 1)}
                if src.get("plan_code") == exp["plan_code"] and pages & wanted:
                    terms = it.get("required_terms", [])
                    if all(norm(t) in norm(text) for t in terms):
                        hit = True
    if it.get("member_id"):
        key = c.execute("SELECT patient::text FROM synthea.enrollment WHERE member_id = %s LIMIT 1", (it["member_id"],)).fetchone()
        for ch in chunks:
            mk = ch["source"].get("member_key")
            if mk and key and str(mk) != key[0]:
                leak = True
    out[it["id"]] = {"pass": bool(ok and hit and not leak), "status": built["status"], "hit": hit, "leak": leak,
                     "confidence": built.get("confidence")}
print(json.dumps(out))
'''


def report(conn: psycopg.Connection, runs: dict) -> str:
    """runs: {(key, reid): run_id}. A table per candidate."""
    lines = []
    for (key, reid), run_id in runs.items():
        g = gate_summary(conn, run_id); t = derive_thresholds(conn, run_id); r = ranking_metrics(conn, run_id)
        lines.append(f"{key}{' +reid' if reid else ''}  run {run_id}: hit@5={g['hit@5']} p@5={g['precision@5']} cov={g['source_coverage']} "
                     f"routing={g['routing_accuracy']} recall={g['complete_recall']} deny={g['deny_clean']} allow={g['allow_answered']} "
                     f"| ties={r['tie_rate']} margin={r['median_margin']} | thresholds prose={t['prose']} record={t['record']} "
                     f"(sep {t['separation']})")
    return "\n".join(lines)
