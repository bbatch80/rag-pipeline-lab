"""Internal-tier documents for ingest: file -> (metadata, backend).

Which directory holds which source is the only thing this module knows;
doc_type, tier, and everything else about a source comes from its
`sources` row. Backend selection by format — Markdown, CSV, and PDF all
normalize to the same Element stream."""

from dataclasses import dataclass
from pathlib import Path

from raglab.metadata import DocumentMeta, derive_internal_meta
from raglab.sources import Registry
from raglab.synth.internal_docs import ALL_DOCS, INTERNAL_DIR

NOTES_DIR = INTERNAL_DIR / "notes"
NOTES_PDF_DIR = INTERNAL_DIR / "notes_pdf"


@dataclass(frozen=True)
class InternalItem:
    path: Path
    meta: DocumentMeta
    backend_kind: str  # markdown | csv | pdf


def items(registry: Registry) -> list[InternalItem]:
    out = []
    for doc in ALL_DOCS:
        path = INTERNAL_DIR / doc.relpath
        if not path.exists():
            continue  # churn may have deleted it — sync handles the rest
        source = registry.for_doc_type(doc.doc_type)
        kind = "csv" if path.suffix == ".csv" else "markdown"
        out.append(
            InternalItem(
                path=path,
                meta=derive_internal_meta(doc.title, source.doc_type, source.acl_tag),
                backend_kind=kind,
            )
        )
    notes = registry["clinical_notes"]
    for directory, kind in ((NOTES_DIR, "markdown"), (NOTES_PDF_DIR, "pdf")):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md" if kind == "markdown" else "*.pdf")):
            out.append(
                InternalItem(
                    path=path,
                    meta=derive_internal_meta(path.stem, notes.doc_type, notes.acl_tag),
                    backend_kind=kind,
                )
            )
    return out
