"""Metadata derivation: from corpus definition + document structure,
never hand-tagging."""

from dataclasses import dataclass

from raglab.chunking import Chunk
from raglab.corpus import CorpusCell


@dataclass(frozen=True)
class DocumentMeta:
    carrier: str
    plan_code: str
    plan_options: tuple[str, ...]
    year: int
    doc_type: str
    acl_tag: str
    effective_date: str
    title: str


def derive_document_meta(cell: CorpusCell) -> DocumentMeta:
    return DocumentMeta(
        carrier=cell.spec.carrier,
        plan_code=cell.spec.plan_code,
        plan_options=cell.spec.options,
        year=cell.year,
        doc_type="brochure",
        acl_tag="public",
        effective_date=f"{cell.year}-01-01",
        title=f"{cell.spec.carrier} {cell.spec.ri} ({', '.join(cell.spec.options)}) {cell.year}",
    )


def chunk_jsonb(doc: DocumentMeta, chunk: Chunk) -> dict:
    """The JSONB payload: descriptive fields. Load-bearing fields
    (year, plan_code, acl_tag) ride as typed columns instead."""
    return {
        "carrier": doc.carrier,
        "plan_options": list(doc.plan_options),
        "doc_type": doc.doc_type,
        "effective_date": doc.effective_date,
        "section": chunk.section,
        "pages": list(chunk.pages),
        "categories": list(chunk.categories),
    }
