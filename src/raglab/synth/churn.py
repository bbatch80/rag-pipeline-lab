"""Churn simulator: seeded mutations/deletions over the churnable internal
pool (KB articles + formulary updates). Golden-anchored documents are never
touched. Feeds the orchestration phase's sync scenarios."""

import random
import re
from dataclasses import dataclass
from pathlib import Path

from raglab.synth.internal_docs import GOLDEN_ANCHORED, INTERNAL_DIR

POOL_GLOBS = ("kb/*.md", "formulary/*.md")
DEFAULT_RATE = 0.08


@dataclass(frozen=True)
class ChurnAction:
    relpath: str
    action: str  # mutated | deleted


def churn_pool(base_dir: Path = INTERNAL_DIR) -> list[Path]:
    pool = []
    for pattern in POOL_GLOBS:
        for path in sorted(base_dir.glob(pattern)):
            if str(path.relative_to(base_dir)) not in GOLDEN_ANCHORED:
                pool.append(path)
    return pool


def run(
    seed: int, rate: float = DEFAULT_RATE, base_dir: Path = INTERNAL_DIR
) -> list[ChurnAction]:
    rng = random.Random(seed)
    pool = churn_pool(base_dir)
    if not pool:
        return []
    k = max(1, round(rate * len(pool)))
    targets = rng.sample(pool, min(k, len(pool)))

    actions = []
    for path in targets:
        relpath = str(path.relative_to(base_dir))
        if rng.random() < 0.2:
            path.unlink()
            actions.append(ChurnAction(relpath=relpath, action="deleted"))
            continue
        text = path.read_text()
        # Tier flips where present, plus an appended revision line — both
        # deterministic under the seed, both guaranteed to change the hash.
        text = re.sub(
            r"Tier (\d)",
            lambda m: f"Tier {rng.choice([t for t in '123' if t != m.group(1)])}",
            text,
            count=1,
        )
        revision = rng.randrange(10**6)
        text += f"\n\n*Revision {revision}: guidance reviewed and updated.*\n"
        path.write_text(text)
        actions.append(ChurnAction(relpath=relpath, action="mutated"))
    return actions
