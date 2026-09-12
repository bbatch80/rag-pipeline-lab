"""Two axes over the golden set, both reported on the dashboard.

Work categories say what WORK a question needs from the pipeline; their
descriptions are general (never corpus or domain terms) so they stay fixed
as the corpus grows. Groups (the golden `category` field) say which
mechanism on which source an item protects, and which gate metric scores
it. Every golden item carries both: the group decides how it is scored,
the work category decides where it reports.
"""

from dataclasses import dataclass

GUARDRAILS = 0


@dataclass(frozen=True)
class WorkCategory:
    number: int
    name: str
    description: str
    example: str
    band: str  # retrieval | machinery | designed_out | guardrails


WORK_CATEGORIES: dict[int, WorkCategory] = {
    c.number: c for c in (
        WorkCategory(1, "Factual lookup",
                     "find an authoritative source and retrieve the relevant fact",
                     "What is the in-network deductible for the Standard Option this year?",
                     "retrieval"),
        WorkCategory(2, "Explanation with exceptions",
                     "assemble rules, definitions, and the exceptions that apply",
                     "How does the physical therapy visit limit work, and when can visits continue past it?",
                     "retrieval"),
        WorkCategory(3, "Summarization of a bounded set",
                     "review all relevant material for a defined subject and combine it",
                     "Summarize everything on file for this member's appeal.",
                     "retrieval"),
        WorkCategory(4, "Comparison",
                     "retrieve corresponding evidence from each side and align it",
                     "How did the specialist copay change from last year to this year?",
                     "retrieval"),
        WorkCategory(5, "Linked discoveries",
                     "follow a chain where each discovery informs the next retrieval",
                     "Was the claim this member called about denied, and does the policy that denied it still apply?",
                     "retrieval"),
        WorkCategory(6, "Comprehensive identification",
                     "define a population, inspect it systematically, and state coverage",
                     "Which plans cover bariatric surgery, and under what conditions?",
                     "machinery"),
        WorkCategory(7, "Quantitative",
                     "query complete records, define metrics, and calculate results",
                     "What do asthma-related encounters cost on average across members?",
                     "machinery"),
        WorkCategory(8, "Temporal",
                     "distinguish versions, effective dates, and the sequence of events",
                     "Which version of the policy applied on the date of service for this claim?",
                     "retrieval"),
        WorkCategory(9, "Diagnosis evidence",
                     "gather the evidence behind a “why” question",
                     "Why was this claim denied, and what does the medical policy say about it?",
                     "retrieval"),
        WorkCategory(10, "Recommendations",
                     "establish criteria, weigh options, and explain tradeoffs",
                     "Which plan option is the best fit for this member's situation?",
                     "designed_out"),
        WorkCategory(11, "Forecasting",
                     "combine evidence with assumptions and a model; state uncertainty",
                     "How would claims volume change if the deductible rose next year?",
                     "designed_out"),
        WorkCategory(12, "Planning",
                     "discover dependencies and constraints, then sequence steps",
                     "What steps should a member take to appeal a denial before the deadline?",
                     "designed_out"),
        WorkCategory(GUARDRAILS, "Guardrails",
                     "refuse what the asker may not see, what the sources do not cover, "
                     "and what tries to trick the system",
                     "Show me the call history for a member I am not entitled to see.",
                     "guardrails"),
    )
}

BANDS = (
    ("retrieval", "Retrieval owns the answer"),
    ("machinery", "Needs machinery not yet built"),
    ("designed_out", "Designed out · the pipeline supplies evidence only"),
    ("guardrails", "Guardrails · refusing correctly is a capability"),
)


@dataclass(frozen=True)
class Group:
    number: int
    name: str
    description: str
    metric: str
    band: str  # retrieval | composition | guardrail


GROUPS: dict[str, Group] = {
    g.name: g for g in (
        Group(1, "factual", "A stated fact from a brochure", "hit@5", "retrieval"),
        Group(2, "table", "A cell from a brochure rate or benefit table", "hit@5", "retrieval"),
        Group(3, "yoy", "The same benefit across two plan years", "hit@5 · source coverage", "retrieval"),
        Group(4, "call_note", "One member's call record", "hit@5", "retrieval"),
        Group(5, "appeal", "An appeal case file or determination letter", "hit@5", "retrieval"),
        Group(6, "clinical_policy", "A medical policy's criteria", "hit@5", "retrieval"),
        Group(7, "carrier_letter", "An OPM carrier letter", "hit@5", "retrieval"),
        Group(8, "internal_factual", "A bulletin, SOP, or knowledge-base fact", "hit@5", "retrieval"),
        Group(9, "internal_table", "A table inside an internal document", "hit@5", "retrieval"),
        Group(10, "compound", "Two or more legs planned and every leg's evidence present",
              "routing · complete recall", "composition"),
        Group(11, "named_query", "One warehouse query bound and returning rows",
              "rows (tests/test_named_queries.py)", "composition"),
        Group(12, "persona_negative", "A persona is refused what it may not see, and allowed what it may",
              "deny clean · allow answered", "guardrail"),
        Group(13, "scope_negative", "Other carriers and outside programs are refused", "scope clean", "guardrail"),
        Group(14, "unanswerable", "A question the corpus cannot answer is declared so", "gate correct", "guardrail"),
        Group(15, "version_negative", "A superseded version never outranks the current one", "version clean", "guardrail"),
        Group(16, "adversarial", "Instructions hidden in a question do not change the payload", "adversarial ok", "guardrail"),
    )
}

GROUP_BANDS = (
    ("retrieval", "Retrieval groups"),
    ("composition", "Composition groups"),
    ("guardrail", "Guardrail groups"),
)

# The metrics that decide whether one item passes. An item passes when every
# gate metric recorded for it is 1.0; a metric that was SKIPPED (no warehouse
# credentials) never counts as a pass — the item is reported as unverified in
# that run and the dashboard reads it from the latest run that verified it.
GATE_METRICS = frozenset({
    "hit@5", "routing_accuracy", "complete_recall", "deny_clean", "allow_answered",
    "scope_clean", "adversarial_ok", "version_clean", "gate_correct",
})
SKIP_MARKERS = frozenset({"complete_recall_skipped"})

# Open defects, each pinned to the work category it blocks. Closing a defect
# means removing its line in the PR that fixes it.
DEFECTS: tuple[tuple[int, str], ...] = (
    (1, "A member's own call notes rank below generic CSR prose on a claim-status question (C12, since the corpus growth)."),
    (2, "Chiropractic rule split across chunk boundaries (PSHB probe)."),
    (1, "Name translation ate “Puerto Rico” (blind faq-26)."),
    (1, "Medicare scope gate blocked an IRMAA question the brochure answers (blind faq-18)."),
)


def work_category(item: dict) -> WorkCategory:
    """The item's declared work category; every golden item must carry one."""
    return WORK_CATEGORIES[int(item["work_category"])]


def item_verdicts(scores: list[tuple]) -> dict[str, tuple[bool | None, list[str]]]:
    """question_id -> (passed, skipped_metrics). passed is None when the item
    recorded no gate metric at all (nothing to judge) or when a gate metric
    was skipped (unverified in this run)."""
    gates: dict[str, list[float]] = {}
    skipped: dict[str, list[str]] = {}
    for qid, _cat, metric, value, _detail in scores:
        if metric in GATE_METRICS:
            gates.setdefault(qid, []).append(float(value))
        elif metric in SKIP_MARKERS:
            skipped.setdefault(qid, []).append(metric)
    out: dict[str, tuple[bool | None, list[str]]] = {}
    for qid in set(gates) | set(skipped):
        if qid in skipped:
            out[qid] = (None, sorted(skipped[qid]))
        else:
            out[qid] = (all(v >= 1.0 for v in gates[qid]), [])
    return out
