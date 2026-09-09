"""Synthetic-tier contract tests: manifest ground truth, referential
integrity, churn determinism, golden-anchor exemption."""

import json
import shutil

import pytest

from raglab.synth import churn, internal_docs, notes

POOL = ("kb/*.md", "formulary/*.md")  # what the sources table flags churn-eligible


@pytest.fixture
def synthea_fixture(db, tmp_path):
    """Minimal synthea schema with a few linked patients, in-transaction."""
    db.execute((notes.config.REPO_ROOT / "db" / "synthea.sql").read_text())
    for i in range(3):
        pid = f"p{i}"
        db.execute(
            "INSERT INTO synthea.patients (id, birthdate, ssn, first, last, address, city) "
            "VALUES (%s, '1980-03-14', '999-11-2222', %s, %s, '12 Elm St', 'Salem')",
            (pid, f"Ana{i}".replace("0", ""), "Pérez"),
        )
        db.execute(
            "INSERT INTO synthea.encounters (id, start, stop, patient) "
            "VALUES (%s, '2024-05-01', '2024-05-03', %s)",
            (f"e{i}", pid),
        )
        db.execute(
            "INSERT INTO synthea.conditions (start, patient, encounter, description) "
            "VALUES ('2024-05-01', %s, %s, 'Essential hypertension')",
            (pid, f"e{i}"),
        )
        db.execute(
            "INSERT INTO synthea.medications (start, patient, description) "
            "VALUES ('2024-05-01', %s, 'lisinopril 10 MG Oral Tablet')",
            (pid,),
        )
    from raglab import enrollment
    enrollment.assign(db)  # generators READ member IDs; the fixture must assign them
    return db


def test_manifest_matches_note_text_exactly(synthea_fixture, tmp_path):
    stats = notes.generate(
        synthea_fixture, count=6, pdf_count=2, seed=7,
        notes_dir=tmp_path / "notes",
        pdf_src_dir=tmp_path / "src",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    assert stats["notes"] == 3  # only 3 patient-condition candidates exist

    entries = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(entries) == stats["notes"]
    for entry in entries:
        stem = entry["doc"].rsplit(".", 1)[0]
        candidates = list((tmp_path / "notes").glob(stem + ".md")) + list(
            (tmp_path / "src").glob(stem + ".txt")
        )
        assert len(candidates) == 1, f"note file missing for {entry['doc']}"
        text = candidates[0].read_text()
        for entity in entry["entities"]:
            assert entity["value"] in text, (
                f"manifest entity {entity} not literally present in {entry['doc']}"
            )
        assert entry["entities"], "every note must contain injected PHI"


def test_notes_reference_existing_patients(synthea_fixture, tmp_path):
    notes.generate(
        synthea_fixture, count=6, pdf_count=0, seed=7,
        notes_dir=tmp_path / "notes",
        pdf_src_dir=tmp_path / "src",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    for line in (tmp_path / "manifest.jsonl").read_text().splitlines():
        pid = json.loads(line)["patient_id"]
        exists = synthea_fixture.execute(
            "SELECT 1 FROM synthea.patients WHERE id = %s", (pid,)
        ).fetchone()
        assert exists, f"note references nonexistent patient {pid}"


def test_generated_notes_are_ascii(synthea_fixture, tmp_path):
    # Injection-time normalization: accented fixture name must come out clean.
    notes.generate(
        synthea_fixture, count=3, pdf_count=0, seed=7,
        notes_dir=tmp_path / "notes",
        pdf_src_dir=tmp_path / "src",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    for path in (tmp_path / "notes").glob("*.md"):
        path.read_text().encode("ascii")  # raises if any non-ASCII survived


def _seeded_pool(tmp_path):
    internal_docs.write_all(base_dir=tmp_path)
    return tmp_path


def test_churn_is_deterministic(tmp_path):
    a = _seeded_pool(tmp_path / "a")
    b = _seeded_pool(tmp_path / "b")
    actions_a = churn.run(seed=99, rate=0.5, base_dir=a, pool_globs=POOL)
    actions_b = churn.run(seed=99, rate=0.5, base_dir=b, pool_globs=POOL)
    assert actions_a == actions_b
    # And the resulting file contents are identical too.
    for action in actions_a:
        if action.action == "mutated":
            assert (a / action.relpath).read_text() == (b / action.relpath).read_text()


def test_churn_never_touches_golden_anchored(tmp_path):
    base = _seeded_pool(tmp_path)
    anchored = {rel: (base / rel).read_text() for rel in internal_docs.GOLDEN_ANCHORED}
    churn.run(seed=1, rate=1.0, base_dir=base, pool_globs=POOL)  # churn the ENTIRE pool
    for rel, before in anchored.items():
        assert (base / rel).exists(), f"golden-anchored {rel} was deleted"
        assert (base / rel).read_text() == before, f"golden-anchored {rel} was mutated"


def test_churn_changes_hashes(tmp_path):
    base = _seeded_pool(tmp_path)
    pool = churn.churn_pool(base, POOL)
    before = {p: p.read_text() for p in pool}
    actions = churn.run(seed=5, rate=0.3, base_dir=base, pool_globs=POOL)
    assert actions
    for action in actions:
        path = base / action.relpath
        if action.action == "deleted":
            assert not path.exists()
        else:
            assert path.read_text() != before[path], "mutation must change content"
