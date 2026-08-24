"""Ingest contract tests: idempotency, quarantine, gate behavior."""

from pathlib import Path

from raglab.gates import run_gates
from raglab.ingest import ingest_document, rel_source_path
from raglab.metadata import DocumentMeta
from raglab.parsing.base import Element


META = DocumentMeta(
    carrier="GEHA",
    plan_code="71-006",
    plan_options=("High", "Standard"),
    program="FEHB",
    year=2026,
    doc_type="brochure",
    acl_tag="public",
    effective_date="2026-01-01",
    title="GEHA 71-006 (High, Standard) 2026",
)


class HealthyBackend:
    name = "stub-healthy"

    def parse(self, path: Path) -> list[Element]:
        elements = []
        for i in range(60):
            elements.append(Element(text=f"Section {i}", category="title", page=i))
            elements.append(
                Element(
                    text=f"Benefit {i}: the deductible and out-of-pocket maximum apply. "
                    + "detail " * 90,
                    category="text",
                    page=i,
                )
            )
        return elements


class DegenerateBackend:
    name = "stub-degenerate"

    def parse(self, path: Path) -> list[Element]:
        return [Element(text="lonely fragment", category="text", page=1)]


def _write_pdf(tmp_path: Path, content: bytes = b"%PDF-fake-v1") -> Path:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(content)
    return pdf


def test_gates_reject_degenerate_parse():
    failures = run_gates(DegenerateBackend().parse(Path("x")), "brochure")
    assert failures, "a near-empty parse must trip the gate"
    assert failures[0].gate == "min_chunks"


def test_idempotency_new_skip_reingest(db, tmp_path):
    pdf = _write_pdf(tmp_path)
    assert ingest_document(db, pdf, META, HealthyBackend()) == "ingested"
    assert ingest_document(db, pdf, META, HealthyBackend()) == "skipped"

    first_count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert first_count > 0

    pdf.write_bytes(b"%PDF-fake-v2-changed")
    assert ingest_document(db, pdf, META, HealthyBackend()) == "reingested"
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == first_count


def test_quarantine_blocks_ingest_and_clears_on_success(db, tmp_path):
    pdf = _write_pdf(tmp_path)
    source = rel_source_path(pdf)

    assert ingest_document(db, pdf, META, DegenerateBackend()) == "quarantined"
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    gates = db.execute(
        "SELECT gate FROM quarantine WHERE source_path = %s", (source,)
    ).fetchall()
    assert gates == [("min_chunks",)]

    assert ingest_document(db, pdf, META, HealthyBackend()) == "ingested"
    backlog = db.execute(
        "SELECT count(*) FROM quarantine WHERE source_path = %s", (source,)
    ).fetchone()[0]
    assert backlog == 0


def test_failed_update_preserves_prior_good_version(db, tmp_path):
    pdf = _write_pdf(tmp_path)
    assert ingest_document(db, pdf, META, HealthyBackend()) == "ingested"

    pdf.write_bytes(b"%PDF-fake-v2-now-corrupt")
    assert ingest_document(db, pdf, META, DegenerateBackend()) == "quarantined"
    # The previous good version must remain queryable.
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
