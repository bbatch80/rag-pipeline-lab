"""Per-document quality gates. A failed gate quarantines the document —
it is never ingested. Silent zero-element parses are the enemy."""

from dataclasses import dataclass
from statistics import median

from raglab.chunking import Chunk

MIN_CHUNKS = 50
MEDIAN_RANGE = (250, 1600)
# A benefits brochure that never mentions these did not parse correctly.
EXPECTED_TERMS = ("out-of-pocket", "deductible")


@dataclass(frozen=True)
class GateFailure:
    gate: str
    detail: str


def run_gates(chunks: list[Chunk], doc_type: str = "brochure") -> list[GateFailure]:
    failures = []

    if len(chunks) < MIN_CHUNKS:
        failures.append(
            GateFailure("min_chunks", f"{len(chunks)} chunks (< {MIN_CHUNKS})")
        )
        return failures  # size/term checks are meaningless on a degenerate parse

    med = median(len(c.text) for c in chunks)
    low, high = MEDIAN_RANGE
    if not low <= med <= high:
        failures.append(
            GateFailure("median_size", f"median {med:.0f} outside [{low}, {high}]")
        )

    if doc_type == "brochure":
        corpus_text = "\n".join(c.text for c in chunks).lower()
        for term in EXPECTED_TERMS:
            if term not in corpus_text:
                failures.append(
                    GateFailure("expected_section", f"term never appears: {term!r}")
                )

    return failures
