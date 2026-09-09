"""Member identifiers and enrollment on the Synthea population — assigned
deterministically, so every generator and both lanes see the same member ID
for the same person.

Enrollment: one row per member per plan year (2021–2026) while the person
is alive. Line of business follows the warehouse's synthetic split (the
same hash rule as snowlane.LOB_EXPR). Plan mix and a ~4%/year open-season
switch are seeded from the person key. Enrollment codes are GEHA's real
FEHB codes (ending 1 = Self Only, 2 = Self and Family, 3 = Self Plus One);
PSHB uses the same numbers with a P suffix — a stated simplification."""

import hashlib

import psycopg

from raglab import identifiers

YEARS = tuple(range(2021, 2027))
SWITCH_RATE = 0.04
# option -> (FEHB plan_code, PSHB plan_code, code stem); weights = plan mix
OPTIONS = {
    "High":         ("71-006", "71-021", "31", 0.35),
    "Standard":     ("71-006", "71-021", "31", 0.25),
    "HDHP":         ("71-014", "71-026", "34", 0.20),
    "Elevate":      ("71-018", "71-022", "25", 0.10),
    "Elevate Plus": ("71-018", "71-022", "25", 0.10),
}
# code suffix per option within the stem (High 311/312/313, Standard 314/315/316, ...)
_CODE = {"High": "311", "Standard": "314", "HDHP": "341", "Elevate": "254", "Elevate Plus": "251"}
TIERS = (("Self Only", 0.45, 0), ("Self and Family", 0.35, 1), ("Self Plus One", 0.20, 2))


def _u(key: str, salt: str) -> float:
    return int(hashlib.sha256(f"{identifiers.SEED}|{salt}|{key}".encode()).hexdigest()[:12], 16) / 16**12


def _pick(u: float, weighted: list[tuple[str, float]]) -> str:
    acc = 0.0
    for name, w in weighted:
        acc += w
        if u < acc:
            return name
    return weighted[-1][0]


def enrollment_code(option: str, tier: str, lob: str) -> str:
    base = int(_CODE[option]) + {"Self Only": 0, "Self and Family": 1, "Self Plus One": 2}[tier]
    return f"{base}" + ("P" if lob == "PSHB" else "")


def rows_for(person_key: str, birth_year: int, death_year: int | None, pshb: bool) -> list[dict]:
    lob = "PSHB" if pshb else "FEHB"
    mix = [(o, w) for o, (_, _, _, w) in OPTIONS.items()]
    option = _pick(_u(person_key, "plan0"), mix)
    tier = _pick(_u(person_key, "tier"), [(t, w) for t, w, _ in TIERS])
    out = []
    for year in YEARS:
        if birth_year > year or (death_year is not None and death_year < year):
            continue
        if year > YEARS[0] and _u(person_key, f"switch{year}") < SWITCH_RATE:
            option = _pick(_u(person_key, f"plan{year}"), mix)
        fehb_code, pshb_code, _, _ = OPTIONS[option]
        out.append({
            "patient": person_key, "member_id": identifiers.member_id(person_key), "year": year,
            "line_of_business": lob, "plan_code": pshb_code if pshb else fehb_code,
            "plan_option": option, "tier": tier, "enrollment_code": enrollment_code(option, tier, lob),
        })
    return out


def assign(conn: psycopg.Connection) -> dict:
    """Fill member_id + mrn on every patient and rebuild enrollment. Idempotent."""
    patients = conn.execute(
        "SELECT id, extract(year FROM birthdate)::int, extract(year FROM deathdate)::int, "
        "abs(hashtext(id)) % 5 = 0 FROM synthea.patients ORDER BY id"
    ).fetchall()
    # Hash-derived values collide occasionally in a 7-/8-digit space; resolve
    # deterministically in sorted patient order by re-hashing with a counter.
    assigned = {}
    taken_m, taken_r = set(), set()
    for p, *_ in patients:
        m = r = None
        for attempt in range(100):
            cand = identifiers.member_id(p, attempt)
            if cand not in taken_m:
                m = cand
                break
        for attempt in range(100):
            cand = identifiers.mrn(p, attempt)
            if cand not in taken_r:
                r = cand
                break
        taken_m.add(m); taken_r.add(r)
        assigned[p] = (m, r)
    with conn.cursor() as cur:
        cur.execute("UPDATE synthea.patients SET member_id = NULL, mrn = NULL")
        cur.executemany(
            "UPDATE synthea.patients SET member_id = %s, mrn = %s WHERE id = %s",
            [(m, r, p) for p, (m, r) in assigned.items()],
        )
        cur.execute("DELETE FROM synthea.enrollment")
        rows = [dict(row, member_id=assigned[p][0])
                for p, by, dy, pshb in patients for row in rows_for(p, by, dy, pshb)]
        cur.executemany(
            "INSERT INTO synthea.enrollment (patient, member_id, year, line_of_business, plan_code, "
            "plan_option, tier, enrollment_code) VALUES (%(patient)s, %(member_id)s, %(year)s, "
            "%(line_of_business)s, %(plan_code)s, %(plan_option)s, %(tier)s, %(enrollment_code)s)",
            rows,
        )
    return {"patients": len(patients), "enrollment_rows": len(rows)}
