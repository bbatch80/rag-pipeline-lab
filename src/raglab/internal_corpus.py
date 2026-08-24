"""Registry of internal-tier documents for ingest: file -> (metadata, backend).

Backend selection by format — Markdown, CSV, and PDF all normalize to the
same Element stream."""

from dataclasses import dataclass
from pathlib import Path

from raglab.metadata import DocumentMeta, derive_internal_meta
from raglab.synth.internal_docs import ALL_DOCS, INTERNAL_DIR

NOTES_DIR = INTERNAL_DIR / "notes"
NOTES_PDF_DIR = INTERNAL_DIR / "notes_pdf"


@dataclass(frozen=True)
class InternalItem:
    path: Path
    meta: DocumentMeta
    backend_kind: str  # markdown | csv | pdf


def items() -> list[InternalItem]:
    out = []
    for doc in ALL_DOCS:
        path = INTERNAL_DIR / doc.relpath
        if not path.exists():
            continue  # churn may have deleted it — sync handles the rest
        kind = "csv" if path.suffix == ".csv" else "markdown"
        out.append(
            InternalItem(
                path=path,
                meta=derive_internal_meta(doc.title, doc.doc_type, doc.acl_tag),
                backend_kind=kind,
            )
        )
    if NOTES_DIR.exists():
        for path in sorted(NOTES_DIR.glob("*.md")):
            out.append(
                InternalItem(
                    path=path,
                    meta=derive_internal_meta(path.stem, "clinical_note", "care_team"),
                    backend_kind="markdown",
                )
            )
    if NOTES_PDF_DIR.exists():
        for path in sorted(NOTES_PDF_DIR.glob("*.pdf")):
            out.append(
                InternalItem(
                    path=path,
                    meta=derive_internal_meta(path.stem, "clinical_note", "care_team"),
                    backend_kind="pdf",
                )
            )
    return out
