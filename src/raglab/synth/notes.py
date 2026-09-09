"""Synthetic clinical notes about real Synthea patients.

Real EHR notes are heavily templated (copy-paste boilerplate, structured
shorthand); these templates mirror that. Every piece of PHI enters the text
through NoteBuilder.inject(), which records it — the manifest is therefore
labeled ground truth for the de-identification benchmark, by construction.

Messiness is deliberate and lives in the text: varied date formats, names
mid-sentence without honorifics, nonstandard member-ID shapes, telegraphic
fragments. Deterministic under a fixed seed.
"""

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import psycopg

from raglab import config, identifiers

NOTES_DIR = config.REPO_ROOT / "data" / "internal" / "notes"
PDF_SRC_DIR = config.REPO_ROOT / "data" / "internal" / "notes_pdf_src"
MANIFEST_PATH = config.REPO_ROOT / "data" / "internal" / "manifests" / "clinical_notes.jsonl"

DEFAULT_COUNT = 250
DEFAULT_PDF_COUNT = 40


def clean_text(value: str) -> str:
    """ASCII-normalize (accents removed) ONCE, before injection, so the
    manifest matches the note text exactly — including in PDF renders.
    Keeps digits: street numbers are legitimate."""
    return (
        unicodedata.normalize("NFKD", value or "")
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
    )


def clean_name(value: str) -> str:
    """Names additionally drop any numeric watermark residue (Synthea
    appends digits to names unless configured off — belt and suspenders)."""
    return clean_text(re.sub(r"\d+", "", value or ""))


_AREA_CODES = (816, 913, 785, 417, 573, 636, 314, 620, 316, 402, 515, 312, 214, 303, 602, 202)


@dataclass
class Injection:
    type: str  # name | date | ssn | member_id | address | phone | mrn
    value: str            # exactly as it appears in the text
    canonical: str = ""   # identifiers: the one canonical value behind the surface form


@dataclass
class NoteBuilder:
    rng: random.Random
    parts: list[str] = field(default_factory=list)
    injections: list[Injection] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.parts.append(text)

    def inject(self, phi_type: str, value: str, canonical: str = "") -> str:
        """The chokepoint: PHI enters text only through here."""
        self.injections.append(Injection(type=phi_type, value=value, canonical=canonical))
        return value

    def name(self, first: str, last: str) -> str:
        style = self.rng.choice(["full", "last_first", "first_only", "last_only"])
        if style == "full":
            return self.inject("name", f"{first} {last}")
        if style == "last_first":
            return self.inject("name", f"{last}, {first}")
        if style == "first_only":
            return self.inject("name", first)
        return self.inject("name", last)

    def date(self, d: date) -> str:
        style = self.rng.choice(["iso", "us", "words", "dots", "short"])
        if style == "iso":
            return self.inject("date", d.isoformat())
        if style == "us":
            return self.inject("date", d.strftime("%m/%d/%Y"))
        if style == "words":
            return self.inject("date", d.strftime("%b %d %Y"))
        if style == "dots":
            return self.inject("date", d.strftime("%m.%d.%y"))
        return self.inject("date", d.strftime("%m/%d/%y"))

    def member_id(self, canonical: str) -> str:
        """The patient's ONE member ID (read from the roster), presented the
        way a clinician might type it. Never invented here."""
        style = self.rng.choice(["canonical", "grouped", "spaced", "bare"])
        return self.inject("member_id", identifiers.present("member_id", canonical, style), canonical)

    def phone(self) -> str:
        # Real numbers have real area codes: member phones come from the
        # enrollment record, so the generator draws valid NANP area codes
        # (mostly Kansas City metro, where GEHA is).
        area = self.rng.choice(_AREA_CODES)
        value = f"({area}) {self.rng.randrange(200, 999)}-{self.rng.randrange(10**4):04d}"
        return self.inject("phone", value)

    def mrn(self, canonical: str) -> str:
        style = self.rng.choice(["canonical", "spaced", "bare"])
        return self.inject("mrn", identifiers.present("mrn", canonical, style), canonical)

    def text(self) -> str:
        return "\n".join(self.parts)


_BOILERPLATE = [
    "Patient was counseled regarding medication adherence, diet, and exercise. "
    "All questions answered. Patient verbalized understanding of the plan of care.",
    "Reviewed allergies, medication list reconciled per protocol. "
    "ROS otherwise negative except as noted above.",
    "Follow-up as scheduled or sooner if symptoms worsen. Return precautions "
    "discussed incl. fever >100.4, chest pain, SOB, or new neuro deficits.",
    "Time spent: greater than 50% of this encounter was spent in counseling "
    "and coordination of care.",
]

_SHORTHAND = {
    "hypertension": "HTN",
    "diabetes": "DM2",
    "shortness of breath": "SOB",
    "complains of": "c/o",
    "history of": "hx",
    "follow up": "f/u",
}


def _soap_note(b: NoteBuilder, p: dict, cond: dict, meds: list[dict], enc: dict):
    b.add(f"CLINIC PROGRESS NOTE - {b.date(enc['start'])}")
    b.add(f"pt: {b.name(p['first'], p['last'])}  DOB {b.date(p['birthdate'])}  "
          f"{b.mrn(p['mrn'])}  member {b.member_id(p['member_id'])}")
    b.add("")
    b.add(f"S: pt c/o sx related to {cond['description'].lower()}, hx as documented. "
          f"{b.rng.choice(['Denies fever/chills.', 'Reports gradual onset.', 'Sx stable since last visit.'])}")
    b.add(f"O: VS stable. Exam notable for findings consistent w/ {cond['description'].lower()}. "
          "No acute distress.")
    med_line = "; ".join(m["description"] for m in meds[:3]) if meds else "no active meds"
    b.add(f"A: {cond['description']}. Current meds: {med_line}.")
    b.add(f"P: continue current regimen, f/u 3 mo. "
          f"Contact office at {b.phone()} with concerns.")
    b.add("")
    b.add(b.rng.choice(_BOILERPLATE))


def _discharge_summary(b: NoteBuilder, p: dict, cond: dict, meds: list[dict], enc: dict):
    b.add("DISCHARGE SUMMARY")
    b.add(f"Patient: {b.name(p['first'], p['last'])}   DOB: {b.date(p['birthdate'])}")
    b.add(f"Member ID {b.member_id(p['member_id'])}   SSN {b.inject('ssn', p['ssn'])}")
    b.add(f"Admit: {b.date(enc['start'])}  Discharge: {b.date(enc['stop'] or enc['start'])}")
    b.add("")
    b.add(f"PRINCIPAL DIAGNOSIS: {cond['description']}")
    b.add("")
    b.add(f"HOSPITAL COURSE: Admitted for management of {cond['description'].lower()}. "
          f"{b.rng.choice(['Course uncomplicated.', 'Improved with treatment.', 'Tolerated interventions well.'])} "
          f"Pt residing at {b.inject('address', p['address'] + ', ' + p['city'])} "
          "with adequate home support.")
    if meds:
        b.add("")
        b.add("DISCHARGE MEDICATIONS:")
        for m in meds[:4]:
            b.add(f"  - {m['description']}")
    b.add("")
    b.add(b.rng.choice(_BOILERPLATE))


def _referral_letter(b: NoteBuilder, p: dict, cond: dict, meds: list[dict], enc: dict):
    b.add(f"Date: {b.date(enc['start'])}")
    b.add("")
    b.add("RE: Specialist referral")
    b.add("")
    b.add(f"Dear colleague, thank you for seeing {b.name(p['first'], p['last'])}, "
          f"DOB {b.date(p['birthdate'])}, member {b.member_id(p['member_id'])}, "
          f"for evaluation of {cond['description'].lower()}.")
    med_line = ", ".join(m["description"] for m in meds[:3]) if meds else "none"
    b.add(f"Relevant meds: {med_line}. Pertinent hx incl. "
          f"{b.rng.choice(['gradual progression', 'suboptimal control', 'new onset sx'])} "
          "despite conservative management.")
    b.add(f"Records available on request - office {b.phone()}. "
          "Appreciate your assessment and recommendations.")
    b.add("")
    b.add(b.rng.choice(_BOILERPLATE))


TEMPLATES = {
    "soap": _soap_note,
    "discharge": _discharge_summary,
    "referral": _referral_letter,
}


def generate(
    conn: psycopg.Connection,
    count: int = DEFAULT_COUNT,
    pdf_count: int = DEFAULT_PDF_COUNT,
    seed: int = 42,
    notes_dir: Path = NOTES_DIR,
    pdf_src_dir: Path = PDF_SRC_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    rng = random.Random(seed)
    candidates = conn.execute(
        """
        SELECT p.id, p.first, p.last, p.birthdate, p.ssn, p.address, p.city,
               c.description AS condition, c.encounter, p.member_id, p.mrn
        FROM synthea.patients p
        JOIN synthea.conditions c ON c.patient = p.id
        WHERE p.first IS NOT NULL AND c.encounter IS NOT NULL
        ORDER BY p.id, c.start
        """
    ).fetchall()
    if not candidates:
        raise RuntimeError("no Synthea patients with conditions — run load-synthea first")

    by_type_counts: dict[str, int] = {}
    notes_dir.mkdir(parents=True, exist_ok=True)
    pdf_src_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    picks = rng.sample(candidates, min(count, len(candidates)))
    with open(manifest_path, "w") as manifest:
        for i, row in enumerate(picks):
            (pid, first, last, birthdate, ssn, address, city,
             condition, encounter_id, member_id, mrn_value) = row
            if not member_id or not mrn_value:
                raise RuntimeError("patients lack member_id/mrn — run `raglab identifiers` first")
            enc = conn.execute(
                "SELECT start, stop FROM synthea.encounters WHERE id = %s",
                (encounter_id,),
            ).fetchone()
            meds = conn.execute(
                """SELECT description FROM synthea.medications
                   WHERE patient = %s ORDER BY start DESC LIMIT 4""",
                (pid,),
            ).fetchall()

            template_name = rng.choice(list(TEMPLATES))
            by_type_counts[template_name] = by_type_counts.get(template_name, 0) + 1
            builder = NoteBuilder(rng=random.Random(rng.randrange(2**32)))
            TEMPLATES[template_name](
                builder,
                {"first": clean_name(first), "last": clean_name(last), "birthdate": birthdate,
                 "ssn": ssn, "address": clean_text(address), "city": clean_text(city),
                 "member_id": member_id, "mrn": mrn_value},
                {"description": condition},
                [{"description": m[0]} for m in meds],
                {"start": enc[0].date() if enc and enc[0] else birthdate,
                 "stop": enc[1].date() if enc and enc[1] else None},
            )

            # First pdf_count notes exist ONLY as PDFs (rendered from a
            # source dir that is never ingested) — no duplicate content.
            if i < pdf_count:
                filename = f"note_{i:04d}_{template_name}.pdf"
                (pdf_src_dir / f"note_{i:04d}_{template_name}.txt").write_text(
                    builder.text()
                )
            else:
                filename = f"note_{i:04d}_{template_name}.md"
                (notes_dir / filename).write_text(builder.text())
            manifest.write(json.dumps({
                "doc": filename,
                "patient_id": pid,
                "template": template_name,
                "entities": [
                    {"type": inj.type, "value": inj.value, "canonical": inj.canonical or inj.value}
                    for inj in builder.injections
                ],
            }) + "\n")

    return {"notes": len(picks), "pdf_bound": min(pdf_count, len(picks)),
            "by_template": by_type_counts}
