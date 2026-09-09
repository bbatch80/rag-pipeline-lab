"""Ingest contract tests: idempotency, quarantine, gate behavior."""

from pathlib import Path

from raglab.gates import GateRules, run_gates
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
    brochure_rules = GateRules(min_chunks=50, median_range=(250, 1600),
                               expected_terms=("out-of-pocket", "deductible"))
    failures = run_gates(DegenerateBackend().parse(Path("x")), brochure_rules)
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


def test_recipe_change_triggers_reingest(db, tmp_path, monkeypatch):
    pdf = _write_pdf(tmp_path)
    monkeypatch.setenv("RAGLAB_CONTEXTUAL", "plain")
    assert ingest_document(db, pdf, META, HealthyBackend()) == "ingested"
    assert ingest_document(db, pdf, META, HealthyBackend()) == "skipped"

    # Same bytes, different processing recipe -> stale, must rebuild.
    monkeypatch.setenv("RAGLAB_CONTEXTUAL", "template")
    assert ingest_document(db, pdf, META, HealthyBackend()) == "reingested"
    assert ingest_document(db, pdf, META, HealthyBackend()) == "skipped"


def test_failed_update_preserves_prior_good_version(db, tmp_path):
    pdf = _write_pdf(tmp_path)
    assert ingest_document(db, pdf, META, HealthyBackend()) == "ingested"

    pdf.write_bytes(b"%PDF-fake-v2-now-corrupt")
    assert ingest_document(db, pdf, META, DegenerateBackend()) == "quarantined"
    # The previous good version must remain queryable.
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0


def test_record_header_carries_the_record_keys_as_vault_tokens(db):
    """Every chunk of a member record gets a search-copy header with the
    record's keys as the vault's pseudonyms — the tokens a translated
    question carries — so section chunks rank on whose record they are."""
    from raglab import deid, ingest

    record = {"case_id": "APL-0038174", "member_id": "M822099594", "claim_id": "CLM-8124637185", "call_id": "C0000105",
              "decision": "upheld"}
    header = ingest.record_header(db, record, "tokenize")
    assert header.startswith("record: case [CASE_ID-") and "member [MEMBER_ID-" in header and "claim [CLAIM_ID-" in header
    assert "call C0000105" in header and "upheld" not in header and "APL-0038174" not in header
    assert header.split("case ")[1].split()[0] == deid._pseudonym(db, "CASE_ID", "APL-0038174")
    assert ingest.record_header(db, {}, "tokenize") == ""
    assert ingest.record_header(db, record, "mask") == "record: case [CASE_ID]  member [MEMBER_ID]  claim [CLAIM_ID]  call C0000105"
    assert ingest.processing_recipe("fast", phi=True, record=True).endswith("|" + ingest.RECORD_HEADER)
    assert ingest.RECORD_HEADER not in ingest.processing_recipe("fast", phi=True)
