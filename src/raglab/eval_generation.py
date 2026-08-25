"""Tier-2 evaluation: end-to-end generation over the demo subset, dual
generators from identical payloads, cross-family judged. Run at checkpoints
and before demos — not per-push (that is Tier 1's job)."""

from dataclasses import dataclass, field

import psycopg

from raglab import ablation, eval_retrieval, judges, payload, rerank, retrieval, router
from raglab.generators import GENERATORS

# Demo subset: representative behaviors, incl. both abstention traps and the
# mirror twin. ~$0.05 per full run at the cheap-generator tier.
DEMO_QUESTION_IDS = (
    "F1", "F4", "F8",      # factual: dollar, plan-disambiguation, boolean
    "T1", "T7",            # tables: premium grid, extreme cell
    "Y6",                  # YoY dual-year
    "A1", "A5",            # internal tier: SOP fact + Jardiance mirror twin
    "U1", "U6",            # abstentions: Ozempic + Medicare parametric trap
)


@dataclass
class GenerationEvalResult:
    run_id: int
    rows: list = field(default_factory=list)  # per (question, generator) dicts
    by_generator: dict = field(default_factory=dict)


def run(conn: psycopg.Connection, config_label: str = "demo") -> GenerationEvalResult:
    conn.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())
    run_id = conn.execute(
        "INSERT INTO eval_runs (kind, config_label, git_sha) "
        "VALUES ('generation', %s, %s) RETURNING id",
        (config_label, eval_retrieval._git_sha()),
    ).fetchone()[0]

    golden = {g["id"]: g for g in ablation.load_golden()}
    result = GenerationEvalResult(run_id=run_id)
    import json as _json

    for qid in DEMO_QUESTION_IDS:
        item = golden[qid]
        decision = router.route(item["question"])
        if decision.scope == "in_scope":
            candidates = retrieval.search(
                conn, item["question"], retrieval.embed_query(item["question"]),
                decision,
            )
            reranked = rerank.rerank(
                item["question"], candidates, stratify_years=decision.years
            )
        else:
            reranked = []
        pl = payload.build(item["question"], decision, reranked)

        for generator in GENERATORS:
            answer = generator.generate(pl)
            verdict = judges.CROSS_JUDGE[generator.name](item, pl, answer)
            row = {
                "question_id": qid,
                "category": item["category"],
                "generator": generator.name,
                "judge": verdict["judge"],
                "correctness": verdict["correctness"],
                "faithfulness": verdict["faithfulness"],
                "abstained": verdict["abstained"],
                "payload_status": pl["status"],
                "answer": answer,
            }
            result.rows.append(row)
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO eval_scores (run_id, question_id, category, "
                    "metric, value, generator, judge, detail) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (run_id, qid, item["category"], metric, value,
                         generator.name, verdict["judge"],
                         _json.dumps({"payload_status": pl["status"],
                                      "answer": answer[:800]}))
                        for metric, value in (
                            ("correctness", verdict["correctness"]),
                            ("faithfulness", verdict["faithfulness"]),
                            ("abstained", float(verdict["abstained"])),
                        )
                    ],
                )
    conn.commit()

    for generator in GENERATORS:
        rows = [r for r in result.rows if r["generator"] == generator.name]
        result.by_generator[generator.name] = {
            "correctness": round(sum(r["correctness"] for r in rows) / len(rows), 3),
            "faithfulness": round(sum(r["faithfulness"] for r in rows) / len(rows), 3),
        }
    return result
