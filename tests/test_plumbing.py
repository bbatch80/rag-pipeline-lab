"""Phase 1 plumbing: record chunk profile, search copy vs display copy,
member keys, registry-driven source discovery."""

from pathlib import Path

from raglab import internal_corpus, sources
from raglab.chunking import chunk_elements
from raglab.ingest import ingest_document
from raglab.metadata import derive_internal_meta
from raglab.parsing.base import Element


def test_record_profile_is_one_chunk_never_merged():
    elements = [Element(text="CALL 2026-03-04", category="title", page=1),
                Element(text="mbr c/o EOB, adv OON ded applies.", category="text", page=1)]
    record = chunk_elements(elements, profile="record")
    assert len(record) == 1 and record[0].section == "CALL 2026-03-04"
    assert "mbr c/o EOB" in record[0].text
    section = chunk_elements(elements, profile="section")
    assert len(section) >= 1  # the default path still works


def test_record_profile_falls_back_when_oversized():
    elements = [Element(text="x" * 3000, category="text", page=1)]
    assert len(chunk_elements(elements, profile="record")) > 1


class _Backend:
    name = "stub"

    def parse(self, path: Path):
        return [Element(text="Title", category="title", page=1),
                Element(text="A member called about a deductible. " * 12, category="text", page=1)]


def test_ingest_writes_search_copy_and_member_key(db, tmp_path):
    f = tmp_path / "note_9999.md"; f.write_text("x")
    meta = derive_internal_meta("note_9999", "clinical_note", "care_team",
                                member_key="11111111-2222-3333-4444-555555555555")
    assert ingest_document(db, f, meta, _Backend()) == "ingested"
    row = db.execute(
        "SELECT c.index_text = c.content, c.member_key::text, d.member_key::text "
        "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.title = 'note_9999'"
    ).fetchone()
    assert row == (True, "11111111-2222-3333-4444-555555555555", "11111111-2222-3333-4444-555555555555")


def test_member_key_backfills_on_skip(db, tmp_path):
    f = tmp_path / "note_9998.md"; f.write_text("x")
    without = derive_internal_meta("note_9998", "clinical_note", "care_team")
    assert ingest_document(db, f, without, _Backend()) == "ingested"
    with_key = derive_internal_meta("note_9998", "clinical_note", "care_team",
                                    member_key="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert ingest_document(db, f, with_key, _Backend()) == "skipped"
    assert db.execute("SELECT member_key::text FROM documents WHERE title = 'note_9998'").fetchone()[0] \
        == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_sources_are_discovered_from_the_registry(db):
    registry = sources.load(db)
    notes = registry["clinical_notes"]
    kinds = {kind for _, kind in internal_corpus.files_for(notes)}
    assert kinds <= {"markdown", "pdf"}
    planned = registry["call_notes"]
    assert internal_corpus.files_for(planned) == [] or planned.status == "ingested"
    for item in internal_corpus.items(registry):
        assert str(item.path).startswith(str(internal_corpus.INTERNAL_DIR))
