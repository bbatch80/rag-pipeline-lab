"""Metadata derivation: from corpus definition + document structure,
never hand-tagging."""

from dataclasses import dataclass

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


def derive_internal_meta(
    title: str, doc_type: str, acl_tag: str, year: int = 2026
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


def chunk_jsonb(doc: DocumentMeta, chunk: Chunk) -> dict:
    """The JSONB payload: descriptive fields. Load-bearing fields
    (year, plan_code, acl_tag) ride as typed columns instead."""
    return {
        "carrier": doc.carrier,
        "program": doc.program,
        "plan_options": list(doc.plan_options),
        "doc_type": doc.doc_type,
        "effective_date": doc.effective_date,
        "section": chunk.section,
        "pages": list(chunk.pages),
        "categories": list(chunk.categories),
    }
