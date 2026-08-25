"""Retrieval ablation over the golden set: vector-only vs BM25-only vs RRF
vs RRF+rerank, measured as hit@k against user-verified source labels.

Relevance: a candidate chunk is relevant to a question if it comes from an
expected source — brochure sources match on (plan_code, year) plus page
overlap (golden pages are printed page numbers; parser pages are PDF pages,
offset +2 on every GEHA brochure, tolerance ±1); internal sources match on
source_path suffix.
"""

import json
from dataclasses import dataclass, field

import psycopg

from raglab import config, rerank, retrieval, router

GOLDEN_PATH = config.REPO_ROOT / "eval" / "golden.jsonl"
PRINTED_TO_PDF_OFFSET = 2
PAGE_TOLERANCE = 1
HIT_K = 5


def load_golden() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]


def is_relevant(candidate: retrieval.Candidate, sources: list[dict]) -> bool:
    for source in sources:
        if "internal" in source:
            if candidate.source_path.endswith(source["internal"]):
                return True
            continue
        if candidate.plan_code != source["plan_code"]:
            continue
        if candidate.year != source["year"]:
            continue
        expected_pdf_pages = {
            p + PRINTED_TO_PDF_OFFSET + delta
            for p in source["printed_pages"]
            for delta in range(-PAGE_TOLERANCE, PAGE_TOLERANCE + 1)
        }
        if expected_pdf_pages & set(candidate.pages):
            return True
    return False


@dataclass
class ArmResult:
    hits: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class AblationReport:
    arms: dict = field(default_factory=dict)  # arm -> ArmResult
    per_question: list = field(default_factory=list)
    answerable_best_scores: list = field(default_factory=list)
    unanswerable_best_scores: list = field(default_factory=list)
    gate_results: list = field(default_factory=list)  # (id, expected, actual)


def _arm_orderings(candidates: list[retrieval.Candidate]) -> dict:
    by_vec = sorted(
        (c for c in candidates if c.vector_rank is not None),
        key=lambda c: c.vector_rank,
    )
    by_txt = sorted(
        (c for c in candidates if c.text_rank is not None),
        key=lambda c: c.text_rank,
    )
    by_rrf = sorted(candidates, key=lambda c: -c.rrf_score)
    return {"vector": by_vec, "bm25": by_txt, "rrf": by_rrf}


def run(conn: psycopg.Connection, k: int = HIT_K) -> AblationReport:
    report = AblationReport(
        arms={a: ArmResult() for a in ("vector", "bm25", "rrf", "rrf+rerank")}
    )
    for item in load_golden():
        if item["category"] == "persona_negative":
            continue  # entitlement assertions live in eval_retrieval; arms run as admin
        decision = router.route(item["question"])
        # Admin sessions are vault-entitled: translate like the pipeline does.
        from raglab import deid

        question = deid.translate_query(conn, item["question"])

        if item.get("unanswerable"):
            expected_gate = item["expected_trigger"] == "scope_gate"
            actually_gated = decision.scope != "in_scope"
            report.gate_results.append((item["id"], expected_gate, actually_gated))
            if not actually_gated:
                candidates = retrieval.search(
                    conn, question,
                    retrieval.embed_query(question), decision,
                )
                reranked = rerank.rerank(question, candidates)
                if reranked:
                    report.unanswerable_best_scores.append(
                        (item["id"], reranked[0].rerank_score)
                    )
            continue

        # Wide fused set so each arm's top-5 is measured un-truncated —
        # otherwise arm metrics shift with fusion composition.
        candidates = retrieval.search(
            conn, question, retrieval.embed_query(question),
            decision, fused_limit=200,
        )
        orderings = _arm_orderings(candidates)
        funnel_input = sorted(candidates, key=lambda c: -c.rrf_score)[:50]
        reranked = rerank.rerank(question, funnel_input, top_n=len(funnel_input))
        orderings["rrf+rerank"] = reranked
        if reranked:
            report.answerable_best_scores.append(
                (item["id"], reranked[0].rerank_score)
            )

        row = {"id": item["id"], "category": item["category"]}
        for arm, ordered in orderings.items():
            hit = any(is_relevant(c, item["sources"]) for c in ordered[:k])
            report.arms[arm].hits += int(hit)
            report.arms[arm].total += 1
            row[arm] = hit
        report.per_question.append(row)
    return report


def markdown_table(report: AblationReport, k: int = HIT_K) -> str:
    lines = [
        f"| arm | hit@{k} |",
        "|---|---:|",
    ]
    for arm, result in report.arms.items():
        lines.append(f"| {arm} | {result.rate:.3f} ({result.hits}/{result.total}) |")
    return "\n".join(lines)
