"""Claim adjudication + appeals: an ADDITIVE pass over the merged records.

The call-note storylines are already generated and merged, and golden
questions name specific notes. So this pass never regenerates them: it
READS the call log and the claim lines and writes the downstream facts —
which is also how a payer's appeals process works.

1. Adjudication overlay: every claim line (encounter) from 2024 on gets a
   status. Where a call note already said what happened to the claim
   ("disputes denial", "claim processed", "pended"), the overlay agrees with
   it — consistency by construction, in the reading direction. Everything
   else is decided by a seeded hash of the encounter id, so adding a
   member or a call never moves another claim's decision.
2. Appeals: cases are drawn from APPEAL_INFO calls whose cited claim is the
   member's own and is denied (~2% of call citations are planted wrong
   claims; those never become appeals — the member would have been told).
   Each case has a member statement, the denial rationale, a clinical
   summary when the denial is clinical (citing one of the member's clinical
   notes when one exists), and a determination — inline, or as a separate
   determination letter rendered to PDF for the first `letters` cases.
3. `appeal_evidence`: what each case cites, for Phase 2's relational
   entitlement.

PHI enters the text only through `inject`, which is what the manifest
records — the de-id eval's ground truth.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import psycopg

from raglab import config, identifiers
from raglab.synth.journeys import CALLS_DIR, _phone
from raglab.synth.notes import clean_name

APPEALS_DIR = config.REPO_ROOT / "data" / "internal" / "appeals"
PDF_SRC_DIR = config.REPO_ROOT / "data" / "internal" / "appeals_pdf_src"
PDF_DIR = config.REPO_ROOT / "data" / "internal" / "appeals_pdf"
MANIFEST_PATH = config.REPO_ROOT / "data" / "internal" / "manifests" / "appeal_documents.jsonl"
CLINICAL_MANIFEST = config.REPO_ROOT / "data" / "internal" / "manifests" / "clinical_notes.jsonl"
END_OF_DATA = date(2026, 8, 31)

# Denial reasons, skewed; the clinical ones cite a medical policy id that
# P1-PR6 (clinical policies) must create.
DENIAL_REASONS = [
    ("eligibility", 0.20), ("prior_auth", 0.18), ("non_covered", 0.17), ("out_of_network", 0.15),
    ("coding", 0.12), ("duplicate", 0.08), ("medical_necessity", 0.10),
]
CLINICAL_REASONS = ("prior_auth", "medical_necessity")
POLICY_COUNT = 20
REASON_TEXT = {
    "eligibility": "the member was not enrolled on the date of service",
    "prior_auth": "the service required prior authorization and none was on file",
    "non_covered": "the service is not a covered benefit under the plan",
    "out_of_network": "the provider is not in the plan's network and the service was not an emergency",
    "coding": "the procedure code submitted is inconsistent with the diagnosis",
    "duplicate": "a claim for the same service on the same date was already processed",
    "medical_necessity": "the service did not meet the plan's medical necessity criteria",
}
BROCHURE_SECTION = {
    "eligibility": "Section 1 (eligibility) and Section 3 (how you get care)",
    "prior_auth": "Section 3 (precertification)",
    "non_covered": "Section 6 (general exclusions)",
    "out_of_network": "Section 5 (network provider cost-sharing)",
    "coding": "Section 7 (filing a claim)",
    "duplicate": "Section 7 (filing a claim)",
    "medical_necessity": "Section 3 (medical necessity)",
}
REVIEWERS = ["RLO", "MDG", "SWH", "JPK", "ABR"]
DECISIONS = [("upheld", 0.55), ("overturned", 0.30), ("partially_overturned", 0.10), ("pending", 0.05)]
RATE_DENIED, RATE_PENDING = 0.09, 0.03


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in (identifiers.SEED, *parts)).encode()).hexdigest()[:12], 16)


def _pick(weights: list[tuple[str, float]], u: float) -> str:
    acc = 0.0
    for name, w in weights:
        acc += w
        if u < acc:
            return name
    return weights[-1][0]


@dataclass
class Entity:
    type: str
    value: str
    canonical: str = ""


@dataclass
class Doc:
    stem: str
    patient: str
    template: str
    year: int
    case_id: str
    record: dict = field(default_factory=dict)  # the case row's required fields (chunk metadata)
    lines: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)

    def inject(self, kind: str, value: str, canonical: str = "") -> str:
        self.entities.append(Entity(kind, value, canonical or value))
        return value

    def text(self) -> str:
        return "\n".join(self.lines)


def _date_str(d: date, rng: random.Random) -> str:
    return rng.choice([d.isoformat(), d.strftime("%m/%d/%Y"), d.strftime("%b %d %Y")])


# ---------------------------------------------------------------- overlay
_HINTS = (("disputes denial", "denied"), ("claim processed", "paid"), ("claim pended", "pending"))


def _call_hints(conn: psycopg.Connection, calls_dir: Path) -> dict[str, str]:
    """claim_id -> status the member was already told, from the note text of
    calls that cite the member's OWN claim (planted wrong citations are
    ignored: that claim belongs to someone else)."""
    # Several calls can cite one claim with different words (the storylines
    # were not sequenced): precedence denied > paid > pending — a disputed
    # denial is a denial; a processed claim is not still pending.
    said: dict[str, set[str]] = {}
    rows = conn.execute(
        "SELECT call_id, patient, claim_id FROM synthea.call_log WHERE claim_id IS NOT NULL ORDER BY call_id"
    ).fetchall()
    owner = dict(conn.execute(
        "SELECT id, patient FROM synthea.encounters WHERE start >= '2024-01-01'"
    ).fetchall())
    claim_owner = {identifiers.claim_id(enc): pat for enc, pat in owner.items()}
    for call_id, patient, claim in rows:
        if claim_owner.get(claim) != patient:
            continue
        path = calls_dir / f"call_{call_id}.md"
        if not path.exists():
            continue
        text = path.read_text().lower()
        for phrase, status in _HINTS:
            if phrase in text:
                said.setdefault(claim, set()).add(status)
    hints: dict[str, str] = {}
    for claim, statuses in said.items():
        hints[claim] = "denied" if "denied" in statuses else "paid" if "paid" in statuses else "pending"
    return hints


def adjudicate(conn: psycopg.Connection, calls_dir: Path = CALLS_DIR) -> dict:
    """Status for every claim line from 2024 on. Deterministic per encounter;
    consistent with what call notes already told the member."""
    hints = _call_hints(conn, calls_dir)
    rows = conn.execute(
        """SELECT e.id, e.patient, p.member_id, e.start::date FROM synthea.encounters e
           JOIN synthea.patients p ON p.id = e.patient
           WHERE e.start >= '2024-01-01' AND p.member_id IS NOT NULL ORDER BY e.id"""
    ).fetchall()
    conn.execute("DELETE FROM appeal_evidence")
    conn.execute("DELETE FROM synthea.appeals")
    conn.execute("DELETE FROM synthea.claim_adjudication")
    out, counts = [], {"paid": 0, "denied": 0, "pending": 0}
    for enc, patient, member_id, start in rows:
        claim = identifiers.claim_id(enc)
        h = _h("adj", enc)
        u = (h % 10_000) / 10_000
        status = hints.get(claim) or ("denied" if u < RATE_DENIED else "pending" if u < RATE_DENIED + RATE_PENDING else "paid")
        if status == "pending" and start < END_OF_DATA - timedelta(days=90) and claim not in hints:
            status = "paid"  # nothing stays pending for years
        decided = start + timedelta(days=10 + (h // 7) % 36) if status != "pending" else None
        if decided and decided > END_OF_DATA:
            decided = END_OF_DATA
        reason = _pick(DENIAL_REASONS, ((h // 13) % 10_000) / 10_000) if status == "denied" else None
        policy = f"CP-{1 + (h // 17) % POLICY_COUNT:04d}" if reason in CLINICAL_REASONS else None
        out.append((enc, claim, patient, member_id, status, decided, reason, policy))
        counts[status] += 1
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO synthea.claim_adjudication (encounter, claim_id, patient, member_id, status, decision_date, denial_reason, policy_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", out,
        )
    counts["hinted_by_calls"] = len(hints)
    return counts


# ---------------------------------------------------------------- appeals
def _clinical_notes_by_patient(path: Path = CLINICAL_MANIFEST) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out.setdefault(rec["patient_id"], []).append(Path(rec["doc"]).stem)
    return out


def _statement(doc: Doc, member: dict, claim_txt: str, dos: str, desc: str, reason: str, call_date: str,
               rng: random.Random) -> list[str]:
    who = doc.inject("name", f"{member['first']} {member['last']}")
    lines = [
        "## Member statement",
        f"I am writing to request reconsideration of the denial of claim {claim_txt} for {desc} on {dos}.",
    ]
    if reason == "prior_auth":
        lines.append(rng.choice([
            "My provider's office told me they had submitted the authorization request before the service.",
            "Nobody told me an authorization was needed; the referral came from my primary care provider.",
        ]))
    elif reason == "out_of_network":
        lines.append("I checked the provider directory before the visit and the provider was listed as participating.")
    elif reason == "medical_necessity":
        lines.append("My physician has documented why this service was needed and I am attaching that record.")
    elif reason == "eligibility":
        lines.append("I was enrolled on that date; my enrollment did not change until open season.")
    elif reason == "duplicate":
        lines.append("These were two separate visits on the same day, not one visit billed twice.")
    else:
        lines.append("I believe the claim was processed incorrectly and ask that it be reviewed.")
    lines.append(f"I called member services on {call_date} and was told to file a written appeal within six months.")
    if rng.random() < 0.4:
        lines.append(f"Please contact me at {doc.inject('phone', _phone(member['id']))}.")
    if rng.random() < 0.3:
        lines.append(f"Date of birth {doc.inject('date', _date_str(member['birthdate'], rng))}.")
    lines.append(f"Signed, {who}.")
    return lines


def generate(conn: psycopg.Connection, cases: int = 300, letters: int = 40, seed: int = 42,
             appeals_dir: Path = APPEALS_DIR, pdf_src_dir: Path = PDF_SRC_DIR,
             manifest_path: Path = MANIFEST_PATH, calls_dir: Path = CALLS_DIR,
             clinical_manifest: Path = CLINICAL_MANIFEST) -> dict:
    rng = random.Random(seed)
    counts = adjudicate(conn, calls_dir)
    candidates = conn.execute(
        """SELECT l.call_id, l.patient, l.member_id, l.call_date, l.claim_id,
                  a.decision_date, a.denial_reason, a.policy_id, a.encounter,
                  e.start::date, e.description, e.total_claim_cost,
                  p.first, p.last, p.birthdate
           FROM synthea.call_log l
           JOIN synthea.claim_adjudication a ON a.claim_id = l.claim_id AND a.patient = l.patient
           JOIN synthea.encounters e ON e.id = a.encounter
           JOIN synthea.patients p ON p.id = l.patient
           WHERE l.reason_code = 'APPEAL_INFO' AND a.status = 'denied'
           ORDER BY l.call_id"""
    ).fetchall()
    notes_by_patient = _clinical_notes_by_patient(clinical_manifest)
    # Clinical denials whose member has a clinical note on file come first
    # (they are what the appeals tier's relational entitlement is about);
    # the rest of the cases are a seeded sample.
    priority = [r for r in candidates if notes_by_patient.get(r[1])]
    rest = [r for r in candidates if r not in priority]
    picks = priority[: cases // 3] + rng.sample(rest, min(cases - len(priority[: cases // 3]), len(rest)))
    picks.sort(key=lambda r: r[0])
    # Decisions are drawn per case in filing order; letters are rendered for
    # a seeded sample of the DECIDED cases (a pending case has no letter).
    decisions = {}
    case_rng = random.Random(seed + 1)
    for idx, r in enumerate(picks):
        call_date, decided_claim = r[3], r[5]
        filed = max(min(call_date + timedelta(days=case_rng.randint(5, 60)), END_OF_DATA,
                        decided_claim + timedelta(days=180)), call_date)
        # Pending only while the 30-day response window is plausibly open.
        decision = "pending" if filed > END_OF_DATA - timedelta(days=25) else _pick(DECISIONS, case_rng.random())
        if decision == "pending" and filed < END_OF_DATA - timedelta(days=60):
            decision = _pick(DECISIONS[:3], case_rng.random() * 0.95)
        decisions[idx] = (filed, decision)
    decided_idx = [i for i, (_, d) in decisions.items() if d != "pending"]
    letter_ids = set(rng.sample(decided_idx, min(letters, len(decided_idx))))

    appeals_dir.mkdir(parents=True, exist_ok=True)
    pdf_src_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for old in list(appeals_dir.glob("appeal_*.md")) + list(pdf_src_dir.glob("letter_*.txt")) + list(PDF_DIR.glob("letter_*.pdf")):
        old.unlink()

    stats = {"cases": 0, "letters_pdf": 0, "clinical_citations": 0, "policy_citations": 0}
    by_decision: dict[str, int] = {}
    with open(manifest_path, "w") as manifest, conn.cursor() as cur:
        for seq, row in enumerate(picks, start=1):
            (call_id, patient, member_id, call_date, claim, decided_claim, reason, policy, encounter,
             dos, desc, cost, first, last, birthdate) = row
            case = identifiers.case_id(seq)
            member = {"id": patient, "first": clean_name(first), "last": clean_name(last), "birthdate": birthdate}
            filed, decision = decisions[seq - 1]
            appeal_type = "pre_service" if reason == "prior_auth" else ("grievance" if rng.random() < 0.05 else "post_service")
            decided = None if decision == "pending" else min(filed + timedelta(days=rng.randint(10, 30)), END_OF_DATA)
            reviewer = rng.choice(REVIEWERS)
            # Members attach medical records to appeals of every kind (an
            # out-of-network denial is argued as an emergency, a non-covered
            # one as medically necessary); a note on file is cited when one exists.
            note_stem = rng.choice(notes_by_patient[patient]) if notes_by_patient.get(patient) else None
            has_letter = seq - 1 in letter_ids and decision != "pending"
            desc = (desc or "visit").strip()

            # No identifier in a file name or a heading: titles and section
            # names are metadata the pipeline shows in the clear.
            record = {"case_id": case, "member_id": member_id, "claim_id": claim, "call_id": call_id, "filed_date": filed.isoformat(),
                      "appeal_type": appeal_type, "denial_reason": reason, "decision": decision,
                      "decided_date": decided.isoformat() if decided else None, "reviewer": reviewer, "policy_id": policy}
            doc = Doc(stem=f"appeal_{seq:04d}", patient=patient, template="appeal", year=filed.year, case_id=case, record=record)
            case_txt = doc.inject("case_id", identifiers.present("case_id", case, rng.choice(["canonical", "canonical", "spaced"])), case)
            mid_txt = doc.inject("member_id", identifiers.present("member_id", member_id, rng.choice(["canonical", "grouped"])), member_id)
            claim_txt = doc.inject("claim_id", identifiers.present("claim_id", claim, rng.choice(["canonical", "grouped", "spaced"])), claim)
            filed_txt = doc.inject("date", _date_str(filed, rng))
            dos_txt = doc.inject("date", _date_str(dos, rng))
            call_txt = doc.inject("date", _date_str(call_date, rng))
            doc.lines += [
                "# Appeal case",
                f"Case {case_txt}  member {mid_txt}  claim {claim_txt}  filed {filed_txt}  type {appeal_type.replace('_', '-')}  denial reason {reason.replace('_', ' ')}",
                "",
            ]
            doc.lines += _statement(doc, member, claim_txt, dos_txt, desc, reason, call_txt, rng)
            doc.lines += [
                "",
                "## Denial rationale",
                f"Claim {claim_txt} ({desc}, date of service {dos_txt}, billed ${float(cost or 0):.0f}) was denied on "
                f"{doc.inject('date', _date_str(decided_claim, rng))} because {REASON_TEXT[reason]}.",
                f"Applicable brochure provisions: {BROCHURE_SECTION[reason]}; disputed claims process: Section 8.",
            ]
            if policy:
                doc.lines.append(f"The denial applied medical policy {policy} (criteria in effect on the date of service).")
                stats["policy_citations"] += 1
            if note_stem or reason in CLINICAL_REASONS:
                doc.lines += ["", "## Clinical summary"]
                if note_stem:
                    doc.lines.append(f"Records reviewed: clinical note {note_stem} for the member, submitted with the appeal.")
                    doc.lines.append(f"The note documents the condition and the service ordered; the reviewer compared it with the criteria of {policy}."
                                     if policy else "The note documents the condition treated and the circumstances of the service.")
                    stats["clinical_citations"] += 1
                else:
                    doc.lines.append("Records requested from the treating provider; no clinical note on file at the time of review.")
            doc.lines += ["", "## Determination"]
            if has_letter:
                letter = Doc(stem=f"letter_{seq:04d}", patient=patient, template="letter", year=(decided or filed).year, case_id=case, record=record)
                lname = letter.inject("name", f"{member['first']} {member['last']}")
                lcase = letter.inject("case_id", case, case)
                lmid = letter.inject("member_id", member_id, member_id)
                lclaim = letter.inject("claim_id", claim, claim)
                ldate = letter.inject("date", decided.strftime("%B %d, %Y"))
                letter.lines += [
                    "GEHA APPEALS DETERMINATION",
                    f"Date: {ldate}",
                    f"Member: {lname}    Member ID: {lmid}",
                    f"Case: {lcase}    Claim: {lclaim}",
                    "",
                    f"Dear {lname},",
                    "",
                    f"We have completed our review of your appeal of the denial of claim {lclaim} for {desc}.",
                    _decision_paragraph(decision, reason, policy),
                    "",
                    f"This decision was made by reviewer {reviewer}. Under Section 8 of your plan brochure, if you disagree "
                    "with our decision you may ask the Office of Personnel Management to review it within 90 days of the date of this letter.",
                    "",
                    "Sincerely,",
                    "GEHA Appeals Department",
                ]
                (pdf_src_dir / f"{letter.stem}.txt").write_text(letter.text() + "\n")
                _write_manifest(manifest, letter, f"{letter.stem}.pdf")
                doc.lines.append(f"Decision {decision.replace('_', ' ')} on {doc.inject('date', _date_str(decided, rng))}; see the determination letter on file ({letter.stem}).")
                cur.execute("INSERT INTO appeal_evidence (case_id, document_title, kind) VALUES (%s, %s, %s)",
                            (case, letter.stem, "determination_letter"))
                stats["letters_pdf"] += 1
            elif decision == "pending":
                doc.lines.append(f"Under review; filed {filed_txt}. Response due within 30 days of receipt (Section 8).")
            else:
                doc.lines.append(f"Decision {decision.replace('_', ' ')} on {doc.inject('date', _date_str(decided, rng))} by reviewer {reviewer}.")
                doc.lines.append(_decision_paragraph(decision, reason, policy))
                doc.lines.append("Under Section 8 of the brochure the member may ask OPM to review this decision within 90 days.")

            (appeals_dir / f"{doc.stem}.md").write_text(doc.text() + "\n")
            _write_manifest(manifest, doc, f"{doc.stem}.md")
            cur.execute(
                "INSERT INTO synthea.appeals (case_id, patient, member_id, encounter, claim_id, call_id, filed_date, appeal_type, "
                "denial_reason, decision, decided_date, reviewer, policy_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (case, patient, member_id, encounter, claim, call_id, filed, appeal_type, reason, decision, decided, reviewer, policy),
            )
            evidence = [(case, f"call_{call_id}", "call_note")]
            if note_stem:
                evidence.append((case, note_stem, "clinical_note"))
            if policy:
                evidence.append((case, policy, "clinical_policy"))
            cur.executemany("INSERT INTO appeal_evidence (case_id, document_title, kind) VALUES (%s, %s, %s)", evidence)
            stats["cases"] += 1
            by_decision[decision] = by_decision.get(decision, 0) + 1
    stats.update({f"adjudication_{k}": v for k, v in counts.items()})
    stats["by_decision"] = by_decision
    stats["candidates"] = len(candidates)
    return stats


def _decision_paragraph(decision: str, reason: str, policy: str | None) -> str:
    if decision == "upheld":
        return (f"After review, the denial is upheld: {REASON_TEXT[reason]}"
                + (f", per medical policy {policy}" if policy else "") + ".")
    if decision == "overturned":
        return ("After review, the denial is overturned and the claim has been sent for reprocessing; "
                "the member's statement and the records provided resolved the reason for denial.")
    if decision == "partially_overturned":
        return ("After review, the denial is partially overturned: the covered portion of the service has been "
                "sent for reprocessing; the remainder stays denied as billed.")
    return "The appeal is under review."


def _write_manifest(manifest, doc: Doc, filename: str) -> None:
    manifest.write(json.dumps({
        "doc": filename, "patient_id": doc.patient, "template": doc.template, "year": doc.year,
        **doc.record, "case_id": doc.case_id,
        "entities": [{"type": e.type, "value": e.value, "canonical": e.canonical} for e in doc.entities],
    }) + "\n")


def render_letters(src_dir: Path = PDF_SRC_DIR, out_dir: Path = PDF_DIR) -> int:
    """Determination letters as PDFs (fpdf2, Helvetica; ASCII-only text)."""
    from raglab.synth.render_pdf import render_all

    return render_all(src_dir, out_dir)
