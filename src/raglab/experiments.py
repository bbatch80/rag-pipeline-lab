"""A/B experiment drivers.

Bake-off: re-ingests ONLY the table-heavy documents (the ones behind the
table-category golden questions) with a chosen parser backend — never
vision-parses the full corpus twice. Deleting the document row first forces
re-ingest past the hash check (the bytes didn't change; the parser did).
"""

import psycopg

from raglab import corpus, ingest
from raglab.metadata import derive_document_meta

# The three 2026 brochures behind every table-category golden question.
BAKEOFF_CELLS = {("71-006", 2026), ("71-014", 2026), ("71-018", 2026)}


def bakeoff_reingest(conn: psycopg.Connection, backend_kind: str) -> list[str]:
    if backend_kind == "docling":
        from raglab.parsing.docling_backend import DoclingBackend

        backend = DoclingBackend()
    else:
        from raglab.parsing.unstructured_backend import UnstructuredBackend

        backend = UnstructuredBackend()

    done = []
    for cell in corpus.cells(dev_only=False):
        if (cell.spec.ri, cell.year) not in BAKEOFF_CELLS:
            continue
        source_path = ingest.rel_source_path(cell.pdf_path)
        conn.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))
        action = ingest.ingest_document(
            conn, cell.pdf_path, derive_document_meta(cell), backend
        )
        done.append(f"{cell.spec.ri}/{cell.year}: {action} ({backend.name})")
    conn.commit()
    return done
