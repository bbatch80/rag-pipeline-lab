"""Additive migrations and the fixed source inventory."""

import pytest

from raglab import ingest, migrations, sources


def test_migrations_apply_once(db):
    """Applying again is a no-op; every file is recorded."""
    assert migrations.apply(db) == []  # already applied to this database
    versions = {r[0] for r in db.execute("SELECT version FROM schema_migrations").fetchall()}
    assert versions == {m.version for m in migrations.available()}


def test_sources_is_the_fixed_inventory(db):
    registry = sources.load(db)
    assert len(registry.all) == 18
    vector = [s for s in registry.all if s.lane in ("vector", "both")]
    assert all(s.doc_type and s.acl_tag for s in vector), "vector sources carry doc_type + tier"
    assert registry["clinical_notes"].phi and not registry["brochures"].phi
    assert registry.for_doc_type("brochure").gate_rules().min_chunks == 50
    with pytest.raises(KeyError):
        registry.for_doc_type("something_new")


def test_every_document_is_bound_to_a_source(db):
    with pytest.raises(Exception):  # NOT NULL: a document without a source is rejected
        db.execute(
            "INSERT INTO documents (source_path, title, content_hash) VALUES ('x/a', 'A', 'h')"
        )


def test_recipe_string_unchanged_by_refactor(monkeypatch):
    """The PHI flag now drives the de-id suffix; the string is byte-identical
    to v1's clinical_note branch, so content hashes did not move."""
    monkeypatch.setenv("RAGLAB_DEID", "tokenize")
    monkeypatch.setenv("RAGLAB_CONTEXTUAL", "template")
    plain = ingest.processing_recipe("fast")
    phi = ingest.processing_recipe("fast", phi=True)
    assert phi == plain + "|deid:tokenize"
    assert "|deid:" not in plain
