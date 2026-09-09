"""Call notes: shorthand dictionary, search-copy normalization, near-duplicate
marking, and storyline integrity."""

import json

from raglab import dedup, identifiers, indexcopy, sources
from raglab.synth import journeys, shorthand


def test_expand_is_idempotent_and_whole_token():
    s = "mbr c/o EOB, adv OON ded applies; xfr to PA line. member ID M048217336"
    once = shorthand.expand(s)
    assert "member complains of explanation of benefits" in once
    assert "advised out-of-network deductible" in once and "transferred to prior authorization" in once
    assert shorthand.expand(once) == once
    assert shorthand.expand("embrace") == "embrace"  # 'mbr' inside a word is untouched


def test_dictionary_covers_the_generator_vocabulary():
    inv = set(shorthand.SHORTHAND.values())
    for word in ("member", "advised", "deductible", "explanation of benefits", "prior authorization",
                 "date of service", "transferred", "call back", "out-of-network", "in-network"):
        assert word in inv


def test_search_copy_normalizes_only_normalized_sources(db):
    registry = sources.load(db)
    calls, notes = registry["call_notes"], registry["clinical_notes"]
    mid = identifiers.member_id("k")
    text = f"mbr {identifiers.present('member_id', mid, 'grouped')} c/o ded"
    out = indexcopy.normalize(text, calls)
    assert out.startswith(f"member {mid} complains of deductible")
    assert indexcopy.normalize(text, notes) == text


def test_boilerplate_suppression_by_document_frequency(tmp_path, monkeypatch):
    from types import SimpleNamespace
    for i in range(30):
        (tmp_path / f"call_{i}.md").write_text(f"CALL {i}\nVerified identity via DOB and member ID.\nunique line {i}.\n")
    src = SimpleNamespace(key="call_notes", dir="x")
    monkeypatch.setattr(indexcopy.config, "REPO_ROOT", tmp_path.parent)
    monkeypatch.setattr(indexcopy, "_boilerplate_cache", {})
    src.dir = tmp_path.name
    out = indexcopy.normalize("CALL 3\nVerified identity via DOB and member ID.\nunique line 3.", src)
    assert "Verified identity" not in out and "unique line 3" in out


def test_near_duplicates_point_at_the_original():
    dedup.reset()
    a = ("CALL NOTE C0000001  03/09/2024  rep JLM  reason CLAIM_STATUS\nmbr M048217336 (Stroman)\n"
         "Verified identity via DOB and member ID.\nReviewed HIPAA disclosure.\n"
         "mbr c/o no update on claim CLM-0004183742 DOS 03/04/2024 (Encounter for check up).\n"
         "claim in process, adv 30 day window, cb if no EOB by then.\ndisp: resolved; cb n; 6 min")
    assert dedup.check_and_add("t", 1, a) is None
    assert dedup.check_and_add("t", 2, a + "\n(duplicate entry - system retry)") == 1
    dedup.forget("t", 1)
    assert dedup.check_and_add("t", 9, a) is None, "after forget, the original's copy is new again"
    b = ("CALL NOTE C0000002  05/01/2025  rep DMS  reason BENEFITS\nmbr M048217336\n"
         "Verified identity via DOB and member ID.\nReviewed HIPAA disclosure.\n"
         "member asked about OOPM for High.\nadvd per sec 5 of the 2025 brochure; mbr will review online.\ndisp: resolved; cb n; 4 min")
    assert dedup.check_and_add("t", 3, b) is None
    dedup.reset()


def test_storylines_are_referentially_intact(db, tmp_path):
    db.execute((journeys.config.REPO_ROOT / "db" / "synthea.sql").read_text())
    for i in range(6):
        db.execute("INSERT INTO synthea.patients (id, birthdate, first, last, ssn) VALUES (%s, '1980-01-01', 'Ann', %s, '1')",
                   (f"p{i}", f"Lee{i}"))
        db.execute("INSERT INTO synthea.encounters (id, start, patient, description, total_claim_cost, payer_coverage) "
                   "VALUES (%s, '2025-03-04', %s, 'Encounter for check up', 120, 100)", (f"e{i}", f"p{i}"))
    from raglab import enrollment
    enrollment.assign(db)
    stats = journeys.generate(db, members=6, target_calls=40, seed=3,
                              calls_dir=tmp_path / "calls", manifest_path=tmp_path / "m.jsonl")
    assert stats["members"] <= 6 and stats["calls"] >= 6
    stats2 = journeys.generate(db, members=6, target_calls=40, seed=3,
                               calls_dir=tmp_path / "calls2", manifest_path=tmp_path / "m2.jsonl")
    assert (tmp_path / "m.jsonl").read_text() == (tmp_path / "m2.jsonl").read_text(), "seeded = reproducible"
    rows = db.execute("SELECT call_id, patient, member_id, claim_id FROM synthea.call_log").fetchall()
    patients = {r[0] for r in db.execute("SELECT id FROM synthea.patients").fetchall()}
    assert all(p in patients for _, p, _, _ in rows), "every call belongs to an existing patient"
    assert all(identifiers.canonicalize("member_id", m) == m for _, _, m, _ in rows)
    assert all(c is None or identifiers.canonicalize("claim_id", c) == c for _, _, _, c in rows)
    for line in (tmp_path / "m.jsonl").read_text().splitlines():
        rec = json.loads(line)
        text = (tmp_path / "calls" / rec["doc"]).read_text()
        for ent in rec["entities"]:
            assert ent["value"] in text, (rec["doc"], ent)
