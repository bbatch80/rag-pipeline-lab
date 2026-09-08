"""The fixed source inventory, read from the `sources` table.

One row per data source (18, seeded by migration 001). Code that used to
branch on doc-type string literals asks the row instead: gate rules, the
PHI flag, the churn pool, the directory a source's files live in."""

from dataclasses import dataclass

import psycopg

from raglab.gates import GateRules


@dataclass(frozen=True)
class Source:
    source_id: int
    key: str
    display_name: str
    department: str
    lane: str
    acl_tag: str | None
    doc_type: str | None
    dir: str | None
    parser: str | None
    chunk_profile: str | None
    gate_min_chunks: int
    gate_median_low: int | None
    gate_median_high: int | None
    gate_expected_terms: tuple[str, ...]
    phi: bool
    cadence: str
    churn_eligible: bool
    status: str

    def gate_rules(self) -> GateRules:
        median_range = None
        if self.gate_median_low is not None and self.gate_median_high is not None:
            median_range = (self.gate_median_low, self.gate_median_high)
        return GateRules(
            min_chunks=self.gate_min_chunks,
            median_range=median_range,
            expected_terms=self.gate_expected_terms,
        )


_COLUMNS = (
    "source_id, key, display_name, department, lane, acl_tag, doc_type, dir, parser, "
    "chunk_profile, gate_min_chunks, gate_median_low, gate_median_high, "
    "gate_expected_terms, phi, cadence, churn_eligible, status"
)


class Registry:
    """All sources, addressable by key or by doc_type."""

    def __init__(self, rows: list[Source]):
        self.by_key = {s.key: s for s in rows}
        self.by_doc_type = {s.doc_type: s for s in rows if s.doc_type}
        self.all = tuple(rows)

    def for_doc_type(self, doc_type: str) -> Source:
        try:
            return self.by_doc_type[doc_type]
        except KeyError:
            raise KeyError(f"no source has doc_type {doc_type!r}; sources are fixed") from None

    def __getitem__(self, key: str) -> Source:
        return self.by_key[key]


def load(conn: psycopg.Connection) -> Registry:
    rows = conn.execute(f"SELECT {_COLUMNS} FROM sources ORDER BY source_id").fetchall()
    return Registry([
        Source(*row[:13], tuple(row[13] or ()), *row[14:]) for row in rows
    ])
