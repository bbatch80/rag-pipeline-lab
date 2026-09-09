"""Member identifiers and enrollment are deterministic and shaped right."""

from datetime import date

from raglab import config, enrollment, identifiers


def _synthea(db):
    db.execute((config.REPO_ROOT / "db" / "synthea.sql").read_text())
    for pid, by, dy in (("p-alive", 1980, None), ("p-child", 2024, None), ("p-dead", 1950, 2023)):
        db.execute(
            "INSERT INTO synthea.patients (id, birthdate, deathdate, first, last) "
            "VALUES (%s, %s, %s, 'A', 'B')",
            (pid, date(by, 6, 1), date(dy, 3, 1) if dy else None),
        )


def test_assign_is_deterministic_and_covers_alive_years(db):
    _synthea(db)
    first = enrollment.assign(db)
    rows1 = db.execute("SELECT patient, year, plan_code, enrollment_code, member_id FROM synthea.enrollment ORDER BY 1, 2").fetchall()
    enrollment.assign(db)
    rows2 = db.execute("SELECT patient, year, plan_code, enrollment_code, member_id FROM synthea.enrollment ORDER BY 1, 2").fetchall()
    assert rows1 == rows2 and first["patients"] == 3
    years = {p: [y for pp, y, *_ in rows1 if pp == p] for p in ("p-alive", "p-child", "p-dead")}
    assert years["p-alive"] == list(range(2021, 2027))
    assert years["p-child"] == [2024, 2025, 2026]
    assert years["p-dead"] == [2021, 2022, 2023]
    mid = db.execute("SELECT member_id, mrn FROM synthea.patients WHERE id = 'p-alive'").fetchone()
    assert mid == (identifiers.member_id("p-alive"), identifiers.mrn("p-alive"))
    assert all(r[4] == identifiers.member_id(r[0]) for r in rows1)


def test_collisions_resolve_deterministically(db, monkeypatch):
    _synthea(db)
    monkeypatch.setattr(identifiers, "_digits", lambda kind, key, n: ("0" * n) if "#" not in key else str(hash(key) % 10**n).zfill(n))
    enrollment.assign(db)
    ids = db.execute("SELECT member_id, mrn FROM synthea.patients").fetchall()
    assert len({m for m, _ in ids}) == 3 and len({r for _, r in ids}) == 3


def test_enrollment_codes_follow_the_fehb_convention():
    assert enrollment.enrollment_code("High", "Self Only", "FEHB") == "311"
    assert enrollment.enrollment_code("High", "Self and Family", "FEHB") == "312"
    assert enrollment.enrollment_code("HDHP", "Self Plus One", "PSHB") == "343P"
