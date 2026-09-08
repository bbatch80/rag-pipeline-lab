"""Per-document quality gates. Rules come from the document's `sources` row
(see raglab.sources); a failed gate quarantines the document — it is never
ingested. Silent zero-element parses are the enemy."""

from dataclasses import dataclass
from statistics import median

from raglab.chunking import Chunk


@dataclass(frozen=True)
class GateRules:
    min_chunks: int
    median_range: tuple[int, int] | None = None
    expected_terms: tuple[str, ...] = ()


DEFAULT_RULES = GateRules(min_chunks=1)


@dataclass(frozen=True)
class GateFailure:
    gate: str
    detail: str


def run_gates(chunks: list[Chunk], rules: GateRules = DEFAULT_RULES) -> list[GateFailure]:
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
