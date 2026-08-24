"""Per-document quality gates, parameterized by document type. A failed gate
quarantines the document — it is never ingested. Silent zero-element parses
are the enemy."""

from dataclasses import dataclass
from statistics import median

from raglab.chunking import Chunk


@dataclass(frozen=True)
class GateRules:
    min_chunks: int
    median_range: tuple[int, int] | None = None
    expected_terms: tuple[str, ...] = ()


DOC_TYPE_RULES = {
    # A benefits brochure that never mentions these did not parse correctly.
    "brochure": GateRules(
        min_chunks=50,
        median_range=(250, 1600),
        expected_terms=("out-of-pocket", "deductible"),
    ),
    "sop": GateRules(min_chunks=1, median_range=(60, 1900)),
    "bulletin": GateRules(min_chunks=1, median_range=(60, 1900)),
    "formulary": GateRules(min_chunks=1),
    "kb": GateRules(min_chunks=1),
    "clinical_note": GateRules(min_chunks=1, median_range=(60, 1900)),
    "rates": GateRules(min_chunks=1),
}

_DEFAULT = GateRules(min_chunks=1)


@dataclass(frozen=True)
class GateFailure:
    gate: str
    detail: str


def run_gates(chunks: list[Chunk], doc_type: str = "brochure") -> list[GateFailure]:
    rules = DOC_TYPE_RULES.get(doc_type, _DEFAULT)
    failures = []

    if len(chunks) < rules.min_chunks:
        failures.append(
            GateFailure("min_chunks", f"{len(chunks)} chunks (< {rules.min_chunks})")
        )
        return failures  # size/term checks are meaningless on a degenerate parse

    if rules.median_range is not None:
        med = median(len(c.text) for c in chunks)
        low, high = rules.median_range
        if not low <= med <= high:
            failures.append(
                GateFailure("median_size", f"median {med:.0f} outside [{low}, {high}]")
            )

    if rules.expected_terms:
        corpus_text = "\n".join(c.text for c in chunks).lower()
        for term in rules.expected_terms:
            if term not in corpus_text:
                failures.append(
                    GateFailure("expected_section", f"term never appears: {term!r}")
                )

    return failures
