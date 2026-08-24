"""Corpus definition: which brochures, which years, which subset."""

from dataclasses import dataclass
from pathlib import Path

from raglab import config


@dataclass(frozen=True)
class BrochureSpec:
    ri: str
    options: tuple[str, ...]
    carrier: str = "GEHA"
    program: str = "FEHB"

    @property
    def plan_code(self) -> str:
        return self.ri


FEHB_PLANS = (
    BrochureSpec("71-006", options=("High", "Standard")),
    BrochureSpec("71-014", options=("HDHP",)),
    BrochureSpec("71-018", options=("Elevate", "Elevate Plus")),
)

YEARS = tuple(range(2021, 2027))

# Fast-path subset for dev/CI iteration only — the full corpus is the product.
# Composition covers both variation axes: cross-plan (all plans, 2026) and
# cross-year (71-006 back years).
DEV_SUBSET = frozenset(
    {("71-006", 2024), ("71-006", 2025), ("71-006", 2026), ("71-014", 2026), ("71-018", 2026)}
)


@dataclass(frozen=True)
class CorpusCell:
    spec: BrochureSpec
    year: int

    @property
    def pdf_path(self) -> Path:
        return config.RAW_DIR / str(self.year) / f"{self.spec.ri}.pdf"

    @property
    def listing_path(self) -> Path:
        return config.RAW_DIR / str(self.year) / f"{self.spec.ri}.json"


def cells(dev_only: bool = True) -> list[CorpusCell]:
    return [
        CorpusCell(spec, year)
        for spec in FEHB_PLANS
        for year in YEARS
        if not dev_only or (spec.ri, year) in DEV_SUBSET
    ]
