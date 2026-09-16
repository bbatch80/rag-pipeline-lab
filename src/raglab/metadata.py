"""Metadata derivation: from corpus definition + document structure,
never hand-tagging."""

from dataclasses import dataclass, field

from raglab.chunking import Chunk
from raglab.corpus import CorpusCell


@dataclass(frozen=True)
class DocumentMeta:
    carrier: str
    plan_code: str | None
    plan_options: tuple[str, ...]
    program: str
    year: int
    doc_type: str
    acl_tag: str
    effective_date: str
    title: str
    member_key: str | None = None  # person key for member-scoped documents
    # The record system's required fields for a member record (case id,
    # claim id, call id, dates, codes): stamped on every chunk as metadata
    # for routing and citation — read from the record, never from the text.
    record: dict = field(default_factory=dict)


def derive_internal_meta(
    title: str, doc_type: str, acl_tag: str, year: int = 2026,
    member_key: str | None = None, record: dict | None = None,
) -> DocumentMeta:
    """Internal-tier documents: no plan_code (they span plans), program
    'internal', year = effective plan year of their content."""
    return DocumentMeta(
        carrier="GEHA",
        plan_code=None,
        plan_options=(),
        program="internal",
        year=year,
        doc_type=doc_type,
        acl_tag=acl_tag,
        effective_date=f"{year}-01-01",
        title=title,
        member_key=member_key,
        record=record or {},
    )


def derive_document_meta(cell: CorpusCell) -> DocumentMeta:
    spec = cell.spec
    return DocumentMeta(
        carrier=spec.carrier,
        plan_code=spec.plan_code,
        plan_options=spec.options,
        program=spec.program,
        year=cell.year,
        doc_type="brochure",
        acl_tag="public",
        effective_date=f"{cell.year}-01-01",
        title=f"{spec.carrier} {spec.program} {spec.ri} ({', '.join(spec.options)}) {cell.year}",
    )


# Option names distinctive enough to tag from a bare mention in a heading;
# the common words (High, Standard) need "Option(s)" beside them.
_DISTINCTIVE_OPTIONS = {"Elevate", "Elevate Plus", "HDHP"}


def section_options(section: str, plan_options: tuple[str, ...] | list[str]) -> list[str]:
    """Which of a document's options its section heading names — the
    option-level metadata the router filters on when a question names an
    option (Elevate vs Elevate Plus, High vs Standard). Derived from the
    heading, never from the text; empty = the heading names none (shared
    or unknown), and a single-option document is never tagged."""
    if len(plan_options) < 2 or not section:
        return []
    import re

    heading = section
    found = []
    for name in plan_options:
        pattern = re.escape(name) + (r"\b(?!\s+Plus\b)" if name == "Elevate" else r"\b")
        if not re.search(r"\b" + pattern, heading, re.IGNORECASE):
            continue
        if name not in _DISTINCTIVE_OPTIONS and not re.search(r"\boptions?\b", heading, re.IGNORECASE):
            continue
        found.append(name)
    return found


def chunk_jsonb(doc: DocumentMeta, chunk: Chunk) -> dict:
    """The JSONB payload: descriptive fields. Load-bearing fields
    (year, plan_code, acl_tag, doc_type) ride as typed columns instead."""
    return {
        "carrier": doc.carrier,
        "program": doc.program,
        "plan_options": list(doc.plan_options),
        **({"section_options": section_options(chunk.section, doc.plan_options)} if len(doc.plan_options) >= 2 else {}),
        "effective_date": doc.effective_date,
        "section": chunk.section,
        "pages": list(chunk.pages),
        "categories": list(chunk.categories),
        **({"record": doc.record} if doc.record else {}),
    }
