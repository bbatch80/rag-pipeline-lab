"""Small statistics for eval reporting. No dependencies."""

import math

# Retrieval metrics that are 0/1 per question — a proportion with a binomial
# interval. Fractional metrics (precision@5, source_coverage) are means of
# [0, 1] values and are reported as means only.
BINARY_METRICS = frozenset({
    "hit@5", "gate_correct", "wrong_abstention", "deny_clean", "scope_clean", "version_clean",
    "allow_answered", "allow_hit", "abstained", "shape_accuracy",
    "routing_accuracy", "complete_recall", "widened_rescue", "adversarial_ok",
})


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% at z=1.96). Unlike the
    normal approximation it behaves at p near 0 or 1 and at small n — which
    is exactly where a 43-question golden set lives."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_diff(before: dict[str, float], after: dict[str, float]) -> dict:
    """Per-question comparison of one binary metric across two runs.
    Inputs map question_id -> value. Returns the ids that flipped each way
    and the ids present in only one run."""
    common = sorted(set(before) & set(after))
    return {
        "gained": [q for q in common if before[q] < after[q]],
        "lost": [q for q in common if before[q] > after[q]],
        "only_before": sorted(set(before) - set(after)),
        "only_after": sorted(set(after) - set(before)),
        "n": len(common),
    }
