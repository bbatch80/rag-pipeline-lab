"""The grown internal library: every document carries the year of its
content and a record with an effective window, the series chain properly
(a superseded bulletin ends where its successor begins), and write_all
emits the manifests the ingest reads."""

import json
from collections import Counter

from raglab.synth import internal_docs, internal_series


def _by_relpath():
    return {d.relpath: d for d in internal_docs.ALL_DOCS}


def test_library_size_is_realistic_not_inflated():
    counts = Counter(d.doc_type for d in internal_docs.ALL_DOCS)
    assert counts["bulletin"] == 24 and counts["sop"] == 16 and counts["kb"] == 24
    assert counts["formulary"] == 11 and counts["rates"] == 1
    assert len({d.relpath for d in internal_docs.ALL_DOCS}) == len(internal_docs.ALL_DOCS), "duplicate relpath"


def test_versioned_kinds_carry_an_effective_window():
    for d in internal_docs.ALL_DOCS:
        if d.doc_type in ("sop", "bulletin", "formulary"):
            assert d.record.get("effective_from"), d.relpath
            assert d.record["status"] in ("current", "superseded"), d.relpath
            assert (d.record["status"] == "superseded") == bool(d.record.get("effective_to")), d.relpath
        assert d.year == int(d.record["effective_from"][:4]) if d.doc_type in ("sop", "bulletin", "formulary") else True


def test_bulletin_chains_end_where_the_successor_begins():
    docs = _by_relpath()
    by_id = {d.record["bulletin_id"]: d for d in docs.values() if d.doc_type == "bulletin"}
    chained = 0
    for d in by_id.values():
        older = d.record.get("supersedes")
        if older:
            assert by_id[older].record["effective_to"] == d.record["effective_from"], (older, d.record["bulletin_id"])
            assert by_id[older].record["status"] == "superseded"
            chained += 1
    assert chained >= 8
    assert by_id["2026-005"].record["supersedes"] == "2025-003" and by_id["2025-003"].record["supersedes"] == "2024-001"


def test_sop_archive_pairs_with_the_current_version():
    docs = _by_relpath()
    for stem in ("sop_deductible_verification", "sop_prior_authorization", "sop_claims_escalation", "sop_cob_medicare"):
        archived, current = docs[f"sops/{stem}_v1.md"], docs[f"sops/{stem}.md"]
        assert archived.record["status"] == "superseded" and archived.record["effective_to"] == current.record["effective_from"]
        assert current.record["version"] == 2 and current.record["supersedes"] == f"{stem}_v1"
        assert archived.content != current.content


def test_formulary_is_per_plan_year_with_glp1s_under_prior_auth():
    docs = _by_relpath()
    f25, f26 = docs["formulary/formulary_2025.md"], docs["formulary/formulary_2026.md"]
    assert f25.record["status"] == "superseded" and f25.record["effective_to"] == "2026-01-01"
    assert f26.record["status"] == "current"
    assert "| Ozempic (semaglutide) | GLP-1 receptor agonist | 2 | Yes |" in f26.content
    assert "| Wegovy (semaglutide) | GLP-1 receptor agonist (weight) | 3 | Yes |" in f25.content
    assert "| Wegovy (semaglutide) | GLP-1 receptor agonist (weight) | 2 | Yes |" in f26.content
    assert docs["formulary/formulary_core.md"].record["status"] == "current"  # golden-anchored, untouched


def test_write_all_emits_manifests_the_ingest_reads(tmp_path):
    n = internal_docs.write_all(base_dir=tmp_path)
    assert n == len(internal_docs.ALL_DOCS)
    for key in ("bulletins", "sops", "kb", "formulary"):
        lines = (tmp_path / "manifests" / f"{key}.jsonl").read_text().splitlines()
        recs = [json.loads(line) for line in lines]
        assert recs and all({"doc", "title", "year"} <= set(r) for r in recs)
        for r in recs:
            assert (tmp_path / key / r["doc"]).exists() if key != "sops" else (tmp_path / "sops" / r["doc"]).exists()
    bulletins = [json.loads(line) for line in (tmp_path / "manifests" / "bulletins.jsonl").read_text().splitlines()]
    assert {r["year"] for r in bulletins} == {2024, 2025, 2026}
    assert all("effective_from" in r for r in bulletins)


def test_golden_anchored_documents_are_unchanged_by_the_growth():
    docs = _by_relpath()
    for rel in internal_docs.GOLDEN_ANCHORED:
        assert rel in docs, rel
    assert "Jardiance (empagliflozin) | SGLT2 inhibitor | 2 | Yes" in docs["formulary/formulary_core.md"].content
    assert internal_series.SERIES_DOCS and all(d.relpath not in internal_docs.GOLDEN_ANCHORED for d in internal_series.SERIES_DOCS)


def test_bulletin_number_is_a_record_key_and_bypasses_the_version_window():
    from raglab.retrieval import _ID_TOKEN, _filters
    from raglab.router import Route

    assert _ID_TOKEN.search("What does Claims Bulletin 2026-002 instruct examiners to do?").group(0) == "Bulletin 2026-002"
    route = Route(scope="in_scope", years=(2026,))
    where, params = _filters(route, None, (), (), {"bulletin_id": "2026-002"}, ("bulletin", "sop"), ("bulletin_id",))
    assert "c.metadata->'record'->>%s IS NOT NULL" in where and params.count("bulletin_id") == 3  # record clause ×2 + bypass
    # A policy id names a family of versions: the window still applies.
    where, params = _filters(route, None, (), (), {"policy_id": "CP-0003"}, ("clinical_policy",), ("bulletin_id",))
    assert "IS NOT NULL" not in where and "effective_from" in where
    assert params.count("policy_id") == 2
