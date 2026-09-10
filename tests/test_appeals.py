"""Appeals + adjudication: an additive pass over the merged call log."""

import json

from raglab import identifiers
from raglab.synth import appeals, journeys


def _seed_population(db):
    db.execute((journeys.config.REPO_ROOT / "db" / "synthea.sql").read_text())
    for i in range(8):
        db.execute("INSERT INTO synthea.patients (id, birthdate, first, last, ssn) VALUES (%s, '1980-01-01', 'Ann', %s, '1')",
                   (f"p{i}", f"Lee{i}"))
        for j in range(3):
            db.execute("INSERT INTO synthea.encounters (id, start, patient, description, total_claim_cost, payer_coverage) "
                       "VALUES (%s, %s, %s, 'Encounter for check up', 120, 100)", (f"e{i}{j}", f"2025-0{j + 3}-04", f"p{i}"))
    from raglab import enrollment
    enrollment.assign(db)


def test_appeals_follow_denied_claims_and_preceding_calls(db, tmp_path):
    _seed_population(db)
    journeys.generate(db, members=8, target_calls=120, seed=3, calls_dir=tmp_path / "calls", manifest_path=tmp_path / "calls.jsonl")
    stats = appeals.generate(db, cases=20, letters=3, seed=5, appeals_dir=tmp_path / "appeals",
                             pdf_src_dir=tmp_path / "appeals_pdf_src", manifest_path=tmp_path / "appeals.jsonl",
                             calls_dir=tmp_path / "calls", clinical_manifest=tmp_path / "none.jsonl")
    assert stats["adjudication_denied"] > 0 and stats["cases"] >= 1
    # every claim line from 2024 on has exactly one adjudication row
    assert db.execute("SELECT count(*) FROM synthea.encounters WHERE start >= '2024-01-01'").fetchone()[0] \
        == db.execute("SELECT count(*) FROM synthea.claim_adjudication").fetchone()[0]
    # every appeal: its member's own DENIED claim line, after the call that disputed it
    rows = db.execute(
        """SELECT a.case_id, a.claim_id, adj.status, adj.patient = a.patient, l.reason_code, a.filed_date >= l.call_date,
                  a.decided_date IS NULL OR a.decided_date >= a.filed_date
           FROM synthea.appeals a JOIN synthea.claim_adjudication adj ON adj.encounter = a.encounter
           JOIN synthea.call_log l ON l.call_id = a.call_id"""
    ).fetchall()
    assert rows and all(status == "denied" and own and reason == "APPEAL_INFO" and after and ordered
                        for _, _, status, own, reason, after, ordered in rows)
    assert all(identifiers.canonicalize("case_id", c) == c for c, *_ in rows)
    # what the case cites is recorded; the call it followed always is
    evidence = db.execute("SELECT case_id, kind FROM appeal_evidence").fetchall()
    assert {c for c, k in evidence if k == "call_note"} == {c for c, *_ in rows}
    # the manifest is exact against the text (de-id ground truth)
    for line in (tmp_path / "appeals.jsonl").read_text().splitlines():
        rec = json.loads(line)
        src = (tmp_path / "appeals" / rec["doc"]) if rec["doc"].endswith(".md") else (tmp_path / "appeals_pdf_src" / rec["doc"].replace(".pdf", ".txt"))
        text = src.read_text()
        assert all(ent["value"] in text for ent in rec["entities"]), rec["doc"]
        assert rec["year"] and rec["case_id"] in [c for c, *_ in rows]


def test_adjudication_agrees_with_what_calls_said(db, tmp_path):
    """A note that says the member disputes a denial means that claim IS
    denied in the overlay; 'claim processed' means paid (precedence denied >
    paid > pending when several calls cite one claim)."""
    _seed_population(db)
    journeys.generate(db, members=8, target_calls=120, seed=3, calls_dir=tmp_path / "calls", manifest_path=tmp_path / "calls.jsonl")
    appeals.adjudicate(db, calls_dir=tmp_path / "calls")
    said: dict[tuple[str, str], set[str]] = {}
    for call_id, claim, patient in db.execute("SELECT call_id, claim_id, patient FROM synthea.call_log WHERE claim_id IS NOT NULL").fetchall():
        text = (tmp_path / "calls" / f"call_{call_id}.md").read_text().lower()
        for phrase, status in appeals._HINTS:
            if phrase in text:
                said.setdefault((claim, patient), set()).add(status)
    checked = 0
    for (claim, patient), statuses in said.items():
        row = db.execute("SELECT status, patient FROM synthea.claim_adjudication WHERE claim_id = %s", (claim,)).fetchone()
        if row is None or row[1] != patient:
            continue  # a planted wrong-claim citation: another member's claim
        expected = "denied" if "denied" in statuses else "paid" if "paid" in statuses else "pending"
        assert row[0] == expected, (claim, statuses, row[0])
        checked += 1
    assert checked > 0


def test_adjudication_is_deterministic(db, tmp_path):
    _seed_population(db)
    journeys.generate(db, members=8, target_calls=60, seed=3, calls_dir=tmp_path / "calls", manifest_path=tmp_path / "calls.jsonl")
    appeals.adjudicate(db, calls_dir=tmp_path / "calls")
    first = db.execute("SELECT encounter, status, denial_reason FROM synthea.claim_adjudication ORDER BY encounter").fetchall()
    appeals.adjudicate(db, calls_dir=tmp_path / "calls")
    assert db.execute("SELECT encounter, status, denial_reason FROM synthea.claim_adjudication ORDER BY encounter").fetchall() == first


def test_out_of_network_denials_require_an_out_of_network_provider(db, tmp_path):
    """Consistency by construction: with a roster and enrollment in place, an
    out-of-network denial never lands on an in-network provider."""
    from raglab import roster
    from raglab.synthea_load import CSV_DIR

    if not (CSV_DIR / "providers.csv").exists():
        import pytest
        pytest.skip("Synthea CSVs not on this machine")
    _seed_population(db)
    roster.load(db)
    orgs = [r[0] for r in db.execute("SELECT id FROM synthea.organizations ORDER BY id LIMIT 40").fetchall()]
    provs = {o: db.execute("SELECT id FROM synthea.providers WHERE organization = %s LIMIT 1", (o,)).fetchone()[0] for o in orgs}
    for i, (pid, eid) in enumerate(db.execute("SELECT patient, id FROM synthea.encounters ORDER BY id").fetchall()):
        o = orgs[i % len(orgs)]
        db.execute("UPDATE synthea.encounters SET organization = %s, provider = %s WHERE id = %s", (o, provs[o], eid))
    journeys.generate(db, members=8, target_calls=120, seed=3, calls_dir=tmp_path / "calls", manifest_path=tmp_path / "calls.jsonl")
    appeals.adjudicate(db, calls_dir=tmp_path / "calls")
    bad = db.execute(
        """SELECT count(*) FROM synthea.claim_adjudication a
           JOIN synthea.encounters e ON e.id = a.encounter
           JOIN synthea.providers pr ON pr.id = e.provider
           JOIN synthea.enrollment en ON en.patient = a.patient AND en.year = EXTRACT(YEAR FROM e.start)
           JOIN synthea.provider_network n ON n.organization = pr.organization AND n.plan_code = en.plan_code
           WHERE a.denial_reason = 'out_of_network' AND n.in_network"""
    ).fetchone()[0]
    assert bad == 0


def test_letters_render_one_pdf_per_source_text(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    for i in range(3):
        (src / f"letter_{i:04d}.txt").write_text("GEHA APPEALS DETERMINATION\nDate: January 1, 2026\n\nDear Member,\n\nUpheld.\n")
    assert appeals.render_letters(src, tmp_path / "pdf") == 3
    assert sorted(p.name for p in (tmp_path / "pdf").glob("*.pdf")) == ["letter_0000.pdf", "letter_0001.pdf", "letter_0002.pdf"]
