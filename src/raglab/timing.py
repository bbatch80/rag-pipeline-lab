"""Per-stage wall-clock timing for a query. A Stopwatch collects milliseconds
per named stage; the pipeline persists the dict on the disclosure row and the
eval summarizes p50/p95 per stage across the golden set.

Displayed, never CI-gated: shared runners are noisy, non-target hardware.
The budget becomes a real measurement on the Azure VM in Phase 6."""

import math
import platform
import time
from contextlib import contextmanager

# Stated budget for a single-source question, end to end (retrieve + rerank +
# payload + disclose). Compound questions get their own (~2 s) in Phase 3.
BUDGET_P95_MS = 1000
STAGES = ("route", "translate", "embed", "search", "rerank", "payload", "disclose")


class Stopwatch:
    def __init__(self):
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(
                self.timings.get(name, 0.0) + (time.perf_counter() - start) * 1000, 2
            )

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

    def snapshot(self) -> dict:
        """Stage ms + total + host, as stored on the disclosure row."""
        return {**self.timings, "total": self.total_ms(), "host": platform.node()}


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; enough for p50/p95 over a golden set."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return round(ordered[k], 1)
