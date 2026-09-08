"""Churn simulator: seeded mutations/deletions over the churnable internal
pool — the sources flagged `churn_eligible` (callers pass their globs; see
pool_globs_for). Golden-anchored documents are never touched. Feeds the
orchestration phase's sync scenarios."""

import random
import re
from dataclasses import dataclass
from pathlib import Path

from raglab.synth.internal_docs import GOLDEN_ANCHORED, INTERNAL_DIR

DEFAULT_RATE = 0.08


def pool_globs_for(registry) -> tuple[str, ...]:
    """Globs (relative to data/internal) of every churn-eligible source."""
    prefix = str(INTERNAL_DIR.relative_to(INTERNAL_DIR.parent.parent)) + "/"
    globs = []
    for source in registry.all:
        if source.churn_eligible and source.dir and source.dir.startswith(prefix):
            globs.append(source.dir[len(prefix):] + "/*.md")
    return tuple(globs)


@dataclass(frozen=True)
class ChurnAction:
    relpath: str
    action: str  # mutated | deleted


def churn_pool(base_dir: Path, pool_globs: tuple[str, ...]) -> list[Path]:
    pool = []
    for pattern in pool_globs:
        for path in sorted(base_dir.glob(pattern)):
            if str(path.relative_to(base_dir)) not in GOLDEN_ANCHORED:
                pool.append(path)
    return pool


def run(
    seed: int, rate: float, base_dir: Path, pool_globs: tuple[str, ...]
) -> list[ChurnAction]:
    rng = random.Random(seed)
    pool = churn_pool(base_dir, pool_globs)
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
