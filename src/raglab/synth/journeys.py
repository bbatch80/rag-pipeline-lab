"""Per-member storylines that emit call notes (Phase 1) — appeals and
adjudication join the same storylines in the next PR, so what a rep wrote,
what the appeal says, and what the warehouse shows stay consistent by
construction.

Realism, stated: call volume is long-tailed (most members call once or
twice, a few dozens of times); reason codes are skewed toward claim status
and benefits questions; a handful of rep personas have distinct shorthand
and macro habits; identifiers are presented the way people type them;
~2% of claim citations are the wrong claim (a valid number, another
member's); ~1.5% of notes are copy-paste near-duplicates (retried saves).
Simplifications: no IVR/chat channels; one phone per member; no households."""

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import psycopg

from raglab import config, identifiers
from raglab.synth.notes import clean_name
from raglab.synth.shorthand import SHORTHAND

CALLS_DIR = config.REPO_ROOT / "data" / "internal" / "calls"
MANIFEST_PATH = config.REPO_ROOT / "data" / "internal" / "manifests" / "call_notes.jsonl"
_AREA_CODES = (816, 913, 785, 417, 573, 636, 314, 620, 316, 402)

REASONS = [  # (code, weight)
    ("CLAIM_STATUS", 0.33), ("BENEFITS", 0.28), ("EOB", 0.10), ("NETWORK", 0.08),
    ("PRIOR_AUTH", 0.06), ("ID_CARD", 0.06), ("DEMOGRAPHICS", 0.05), ("APPEAL_INFO", 0.04),
]
DISPOSITIONS = ["resolved", "resolved", "resolved", "callback scheduled", "escalated", "info provided", "transferred"]

REPS = [  # initials, style, macros
    ("JLM", "shorthand", ["Verified identity via DOB and member ID.", "Reviewed HIPAA disclosure."]),
    ("RKO", "shorthand", ["ID verified (DOB + member ID).", "Adv call may be recorded."]),
    ("TAB", "plain", ["Verified identity via DOB and member ID.", "Reviewed HIPAA disclosure."]),
    ("DMS", "macro", ["Verified identity via DOB and member ID.", "Reviewed HIPAA disclosure.",
                      "Advised member of privacy practices.", "Offered secure message follow-up."]),
    ("PNV", "plain", ["Identity verified."]),
    ("KRS", "shorthand", ["Verified identity via DOB and member ID."]),
]


@dataclass
class Entity:
    type: str
    value: str
    canonical: str = ""


@dataclass
class Note:
    call_id: str
    patient: str
    member_id: str
    when: date
    rep: str
    reason: str
    disposition: str
    claim_id: str | None
    duration: int
    lines: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    duplicate_of: str | None = None

    def inject(self, kind: str, value: str, canonical: str = "") -> str:
        self.entities.append(Entity(kind, value, canonical or value))
        return value

    def text(self) -> str:
        return "\n".join(self.lines)


def _phone(person_key: str) -> str:
    h = int(hashlib.sha256(f"{identifiers.SEED}|phone|{person_key}".encode()).hexdigest()[:10], 16)
    return f"({_AREA_CODES[h % len(_AREA_CODES)]}) {200 + (h // 7) % 800}-{(h // 5) % 10000:04d}"


def _sh(word: str, style: str, rng: random.Random) -> str:
    """Shorthand-style reps abbreviate (mostly); plain reps write it out."""
    inv = {v: k for k, v in SHORTHAND.items()}
    if word in inv and (style == "shorthand" and rng.random() < 0.85 or style == "macro" and rng.random() < 0.5):
        return inv[word]
    return word


def _date_str(d: date, rng: random.Random) -> str:
    return rng.choice([d.isoformat(), d.strftime("%m/%d/%Y"), d.strftime("%m/%d/%y"), d.strftime("%b %d %Y")])


def _calls_for(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.67:
        return rng.randint(1, 3)
    if u < 0.92:
        return rng.randint(4, 10)
    return rng.randint(15, 45)


def _body(note: Note, thread: str, member: dict, enc: dict | None, plan: dict | None,
          style: str, rng: random.Random, wrong_claim: str | None) -> list[str]:
    s = lambda w: _sh(w, style, rng)  # noqa: E731
    claim = None
    if enc is not None:
        claim = wrong_claim or identifiers.claim_id(enc["id"])
        claim_txt = note.inject("claim_id", identifiers.present("claim_id", claim, rng.choice(["canonical", "grouped", "spaced"])), claim)
        dos = note.inject("date", _date_str(enc["start"], rng))
    opt = plan["plan_option"] if plan else "the plan"
    if thread == "CLAIM_STATUS":
        return [f"{s('member')} {s('complains of')} no update on claim {claim_txt} {s('date of service')} {dos} ({enc['description'][:40]}).",
                rng.choice([f"claim in process, {s('advised')} 30 day window, {s('call back')} if no {s('explanation of benefits')} by then.",
                            f"claim processed {s('effective')} last week; {s('explanation of benefits')} mailed; {s('advised')} allowed {s('amount')} ${enc['cost']:.0f}.",
                            f"claim pended for {s('information')} from provider; {s('advised')} {s('member')} no action needed."])]
    if thread == "BENEFITS":
        q = rng.choice([f"{s('deductible')} for {opt} this {s('calendar year')}", f"{s('out-of-pocket maximum')} for {opt}",
                        f"{s('specialist')} copay {s('in-network')} under {opt}", f"{s('urgent care')} vs {s('emergency room')} cost share on {opt}",
                        f"whether {s('prescription')} tier 2 is {s('covered')} on {opt}"])
        return [f"{s('member')} asked about {q}.",
                f"{s('advised')} per {s('section')} 5 of the {plan['year'] if plan else 2026} brochure; {s('member')} {rng.choice(['satisfied', 'will review online', 'requested mailed copy'])}."]
    if thread == "EOB":
        return [f"{s('member')} confused by {s('explanation of benefits')} for claim {claim_txt} {s('date of service')} {dos}: {s('patient')} responsibility ${max(0, enc['cost'] - enc['coverage']):.0f}.",
                f"{s('advised')} {s('out-of-network')} {s('deductible')} applies; explained {s('balance')} vs plan allowance." if rng.random() < 0.5 else
                f"{s('advised')} {s('in-network')} copay applied correctly; {s('member')} understood."]
    if thread == "NETWORK":
        return [f"{s('member')} asked if new {s('primary care provider')} is {s('in-network')} for {opt}.",
                f"{s('advised')} to use provider directory; {s('confirmed')} {rng.choice(['participating', 'not participating — adv OON cost share'])}."]
    if thread == "PRIOR_AUTH":
        return [f"{s('member')} asked whether {rng.choice(['MRI', 'sleep study', 'infusion', 'inpatient rehab'])} needs {s('prior authorization')}.",
                f"{s('advised')} {s('prior authorization')} required; {s('referral')} from {s('primary care provider')} {s('requested')}; {s('transferred')} to {s('prior authorization')} line."]
    if thread == "ID_CARD":
        return [f"{s('member')} {s('requested')} replacement ID card for {opt}.", f"card {s('requested')}; {s('advised')} 7-10 business days; digital card on portal."]
    if thread == "DEMOGRAPHICS":
        return [f"{s('member')} updated phone to {note.inject('phone', _phone(member['id']))}.",
                f"{s('advised')} change {s('effective')} immediately; {s('confirmed')} {s('enrollment')} unchanged."]
    return [f"{s('member')} disputes denial on claim {claim_txt} {s('date of service')} {dos}.",  # APPEAL_INFO
            f"{s('advised')} appeal rights per {s('section')} 8: written request within 6 months; {s('transferred')} to appeals."]


def generate(conn: psycopg.Connection, members: int = 1500, target_calls: int = 10000, seed: int = 42,
             calls_dir: Path = CALLS_DIR, manifest_path: Path = MANIFEST_PATH) -> dict:
    rng = random.Random(seed)
    people = conn.execute(
        """SELECT p.id, p.first, p.last, p.birthdate, p.member_id, p.mrn
           FROM synthea.patients p
           WHERE p.member_id IS NOT NULL AND p.first IS NOT NULL
             AND EXISTS (SELECT 1 FROM synthea.encounters e WHERE e.patient = p.id AND e.start >= '2024-01-01')
           ORDER BY p.id"""
    ).fetchall()
    if not people:
        raise RuntimeError("no eligible patients — run load-synthea and identifiers first")
    picks = rng.sample(people, min(members, len(people)))
    calls_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for old in calls_dir.glob("call_*.md"):
        old.unlink()
    conn.execute("DELETE FROM synthea.call_log")

    notes: list[Note] = []
    seq = 0
    all_encounters = conn.execute(
        "SELECT id FROM synthea.encounters WHERE start >= '2024-01-01' ORDER BY id LIMIT 5000"
    ).fetchall()
    i = 0
    while len(notes) < target_calls and i < len(picks):
        pid, first, last, birthdate, member_id, mrn = picks[i]; i += 1
        member = {"id": pid, "first": clean_name(first), "last": clean_name(last), "birthdate": birthdate,
                  "member_id": member_id}
        encs = conn.execute(
            """SELECT id, start::date, description, total_claim_cost, payer_coverage FROM synthea.encounters
               WHERE patient = %s AND start >= '2024-01-01' ORDER BY start DESC LIMIT 12""", (pid,)
        ).fetchall()
        encs = [{"id": e[0], "start": e[1], "description": e[2] or "visit", "cost": float(e[3] or 0), "coverage": float(e[4] or 0)} for e in encs]
        plans = {r[0]: {"year": r[0], "plan_code": r[1], "plan_option": r[2]} for r in conn.execute(
            "SELECT year, plan_code, plan_option FROM synthea.enrollment WHERE patient = %s", (pid,)).fetchall()}
        n = _calls_for(rng)
        threads = [rng.choices([c for c, _ in REASONS], [w for _, w in REASONS])[0] for _ in range(n)]
        prev: Note | None = None
        for thread in threads:
            seq += 1
            enc = rng.choice(encs) if encs and thread in ("CLAIM_STATUS", "EOB", "APPEAL_INFO") else None
            when = (enc["start"] + timedelta(days=rng.randint(3, 60))) if enc else date(rng.choice([2024, 2025, 2026]), rng.randint(1, 12), rng.randint(1, 28))
            when = min(when, date(2026, 8, 31))
            plan = plans.get(when.year)
            rep, style, macros = rng.choice(REPS)
            wrong = identifiers.claim_id(rng.choice(all_encounters)[0]) if enc and rng.random() < 0.02 else None
            note = Note(call_id=f"C{seq:07d}", patient=pid, member_id=member_id, when=when, rep=rep, reason=thread,
                        disposition=rng.choice(DISPOSITIONS), claim_id=None, duration=rng.randint(120, 1500))
            mid_style = rng.choice(["canonical", "grouped", "spaced", "bare"])
            header_id = note.inject("member_id", identifiers.present("member_id", member_id, mid_style), member_id)
            note.lines.append(f"CALL NOTE {note.call_id}  {note.inject('date', _date_str(when, rng))}  rep {rep}  reason {thread}")
            who = f"{_sh('member', style, rng)} {header_id}"
            if rng.random() < 0.5:
                who += f" ({note.inject('name', rng.choice([f'{member['first']} {member['last']}', f'{member['last']}, {member['first']}', member['last']]))})"
            if rng.random() < 0.3:
                who += f"  {_sh('date of birth', style, rng)} {note.inject('date', _date_str(birthdate, rng))}"
            if rng.random() < 0.15:
                who += f"  {_sh('MRN', style, rng)} {note.inject('mrn', identifiers.present('mrn', mrn, rng.choice(['canonical', 'spaced'])), mrn)}"
            note.lines.append(who)
            k = len(macros) if style == "macro" else rng.randint(0, min(2, len(macros)))
            note.lines.extend(macros[:k])
            note.lines.extend(_body(note, thread, member, enc, plan, style, rng, wrong))
            if enc:
                note.claim_id = wrong or identifiers.claim_id(enc["id"])
            note.lines.append(f"disp: {note.disposition}; {_sh('call back', style, rng)} {'y' if 'callback' in note.disposition else 'n'}; {note.duration // 60} min")
            notes.append(note)
            # copy-paste near-duplicate (retried save)
            if prev is not None and rng.random() < 0.015:
                seq += 1
                dup = Note(call_id=f"C{seq:07d}", patient=pid, member_id=member_id, when=when, rep=rep, reason=thread,
                           disposition=note.disposition, claim_id=note.claim_id, duration=note.duration,
                           lines=[note.lines[0].replace(note.call_id, f"C{seq:07d}")] + note.lines[1:] + ["(duplicate entry - system retry)"],
                           entities=list(note.entities), duplicate_of=f"call_{note.call_id}.md")
                notes.append(dup)
            prev = note

    with open(manifest_path, "w") as manifest, conn.cursor() as cur:
        for note in notes:
            (calls_dir / f"call_{note.call_id}.md").write_text(note.text() + "\n")
            rec = {"doc": f"call_{note.call_id}.md", "patient_id": note.patient, "template": note.reason,
                   "entities": [{"type": e.type, "value": e.value, "canonical": e.canonical} for e in note.entities]}
            if note.duplicate_of:
                rec["duplicate_of"] = note.duplicate_of
            manifest.write(json.dumps(rec) + "\n")
            cur.execute(
                "INSERT INTO synthea.call_log (call_id, patient, member_id, call_date, rep_id, reason_code, disposition, claim_id, duration_sec) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (note.call_id, note.patient, note.member_id, note.when, note.rep, note.reason, note.disposition, note.claim_id, note.duration),
            )
    by_reason: dict[str, int] = {}
    for n_ in notes:
        by_reason[n_.reason] = by_reason.get(n_.reason, 0) + 1
    per_member = {}
    for n_ in notes:
        per_member[n_.patient] = per_member.get(n_.patient, 0) + 1
    return {"calls": len(notes), "members": len(per_member), "by_reason": by_reason,
            "duplicates": sum(1 for n_ in notes if n_.duplicate_of),
            "max_per_member": max(per_member.values()), "median_per_member": sorted(per_member.values())[len(per_member) // 2]}
