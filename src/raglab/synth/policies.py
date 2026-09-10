"""Clinical (medical) policies: authored, versioned, deterministic.

Twenty policies, CP-0001..CP-0020 — the ids the adjudication overlay and
the appeals already cite. Each states when a service is medically
necessary: purpose, criteria, documentation, exclusions, references. Ten of
them have a second version in which ONE criterion changes, effective during
2025, so a question about that criterion has a different correct answer
under each version and an appeal's date of service selects one of them.

Every version is a document of its own (superseded ones are never deleted:
claims are adjudicated under the version in effect on the date of service).
The version fields — policy id, version, effective window, status,
supersedes — are the record's required fields and ride on every chunk as
metadata; retrieval filters on them (raglab.retrieval, `as_of`). Public
tier: payers publish these. No PHI."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from raglab import config

POLICIES_DIR = config.REPO_ROOT / "data" / "internal" / "policies"
MANIFEST_PATH = config.REPO_ROOT / "data" / "internal" / "manifests" / "clinical_policies.jsonl"
V1_EFFECTIVE = date(2023, 1, 1)


@dataclass(frozen=True)
class Criterion:
    text: str            # with {v} where the value goes
    v1: str
    v2: str | None = None  # None: unchanged in version 2


@dataclass(frozen=True)
class Policy:
    pid: str
    title: str
    service: str
    purpose: str
    criteria: tuple[Criterion, ...]
    documentation: tuple[str, ...]
    exclusions: tuple[str, ...]
    v2_effective: date | None  # None: single version
    brochure_ref: str = "Section 3 (medical necessity and precertification)"


POLICIES: tuple[Policy, ...] = (
    Policy("CP-0001", "Advanced Imaging (MRI and CT)", "MRI or CT imaging",
           "Advanced imaging is covered when a clinical question cannot be answered by history, examination, or plain radiography.",
           (Criterion("Symptoms have persisted for at least {v} despite conservative treatment, or a red-flag finding is documented.", "6 weeks", "4 weeks"),
            Criterion("The ordering provider documents the specific clinical question the study will answer.", "yes"),
            Criterion("Plain radiography has been performed within the prior {v} when relevant to the region imaged.", "90 days")),
           ("Ordering provider's note with the clinical question", "Prior imaging report when applicable", "Conservative treatment history with dates"),
           ("Screening imaging in the absence of symptoms", "Repeat imaging within 30 days without a documented change"),
           date(2025, 4, 1)),
    Policy("CP-0002", "Emergency Room Admission Review", "an emergency room admission",
           "Emergency care is covered when a prudent layperson would expect a delay to endanger life or health; admission from the emergency room is reviewed against inpatient criteria.",
           (Criterion("Presenting symptoms meet the prudent layperson standard as documented at triage.", "yes"),
            Criterion("Inpatient admission from the emergency room requires documented instability or a need for services that cannot be delivered in observation within {v}.", "24 hours", "48 hours"),
            Criterion("Observation status is applied when the expected stay is under {v}.", "two midnights")),
           ("Triage note with vital signs and chief complaint", "Emergency physician's disposition note", "Admission order with level-of-care rationale"),
           ("Scheduled procedures presented through the emergency room", "Care that could have been delivered in urgent care as documented by the treating physician"),
           date(2025, 6, 1)),
    Policy("CP-0003", "Sleep Studies (Polysomnography)", "an attended in-lab sleep study",
           "In-lab polysomnography is covered when a home sleep test is inappropriate or inconclusive.",
           (Criterion("A home sleep apnea test has been attempted and was inconclusive, or the member has a documented comorbidity that makes home testing inappropriate.", "yes"),
            Criterion("Symptoms have been present for at least {v}.", "3 months", "6 weeks"),
            Criterion("Body mass index, neck circumference, and a validated sleepiness score are documented.", "yes")),
           ("Sleep questionnaire and sleepiness score", "Home sleep test result when performed", "Comorbidity documentation when home testing is bypassed"),
           ("Repeat studies within 12 months without a change in clinical status", "Studies ordered for snoring alone"),
           date(2025, 5, 1)),
    Policy("CP-0004", "Home Infusion Therapy", "home infusion therapy",
           "Home infusion is covered when the drug is appropriate for home administration and the home setting is safe.",
           (Criterion("The prescribed drug appears on the plan's home infusion list.", "yes"),
            Criterion("The member or caregiver has completed administration training, or nursing visits are arranged for at least the first {v}.", "3 infusions", "5 infusions"),
            Criterion("A treating physician's order states drug, dose, route, frequency, and duration.", "yes")),
           ("Physician order", "Training attestation or nursing plan", "Pharmacy compounding record when applicable"),
           ("Drugs requiring continuous cardiac monitoring", "First doses of drugs with a documented high anaphylaxis risk"),
           date(2025, 7, 1)),
    Policy("CP-0005", "Inpatient Rehabilitation Facility Admission", "inpatient rehabilitation",
           "Inpatient rehabilitation is covered when the member requires intensive multidisciplinary therapy that cannot be provided at a lower level of care.",
           (Criterion("The member can participate in at least {v} of therapy per day, five days per week.", "3 hours"),
            Criterion("At least two therapy disciplines are required, one of which is physical or occupational therapy.", "yes"),
            Criterion("A physician with rehabilitation training conducts face-to-face visits at least {v}.", "three times per week", "daily")),
           ("Pre-admission screening within 48 hours of admission", "Therapy tolerance assessment", "Physician certification"),
           ("Members who can be safely served by a skilled nursing facility", "Maintenance therapy without measurable goals"),
           date(2025, 3, 1)),
    Policy("CP-0006", "Durable Medical Equipment: Power Mobility Devices", "a power wheelchair or scooter",
           "Power mobility devices are covered when a mobility limitation prevents activities of daily living in the home and cannot be resolved with a cane, walker, or manual wheelchair.",
           (Criterion("A face-to-face mobility examination was performed within {v} of the order.", "45 days", "60 days"),
            Criterion("The member cannot safely use a manual wheelchair, as documented.", "yes"),
            Criterion("The home permits use of the device, as documented by a home assessment.", "yes")),
           ("Mobility examination note", "Home assessment", "Detailed product description signed by the physician"),
           ("Devices requested primarily for use outside the home", "Upgrades for convenience features"),
           date(2025, 8, 1)),
    Policy("CP-0007", "Physical Therapy Visit Limits and Continuation", "continued physical therapy",
           "Physical therapy is covered while measurable functional improvement is documented.",
           (Criterion("An initial evaluation documents functional deficits and measurable goals.", "yes"),
            Criterion("Continuation beyond {v} requires a progress report showing measurable improvement toward the goals.", "12 visits", "20 visits"),
            Criterion("Re-evaluation occurs at least every 30 days.", "yes")),
           ("Initial evaluation with goals", "Progress reports at each continuation point", "Plan of care signed by the treating provider"),
           ("Maintenance programs the member can perform independently", "Duplicate therapy from two providers for the same condition"),
           date(2025, 9, 1)),
    Policy("CP-0008", "Bariatric Surgery", "bariatric surgery",
           "Bariatric surgery is covered for members who meet body mass index and comorbidity thresholds after a structured preparation program.",
           (Criterion("Body mass index of at least {v} (five points lower with an obesity-related comorbidity).", "40", "35"),
            Criterion("Completion of a physician-supervised weight management program of at least {v}.", "6 months", "3 months"),
            Criterion("Psychological evaluation within the prior 12 months finds no contraindication.", "yes")),
           ("Weight management program records with dates", "Psychological evaluation", "Surgical consultation note"),
           ("Procedures not on the plan's covered bariatric procedure list", "Revision surgery for cosmetic reasons"),
           date(2025, 10, 1)),
    Policy("CP-0009", "Genetic Testing for Hereditary Cancer Risk", "hereditary cancer genetic testing",
           "Genetic testing is covered when the result will change medical management and family or personal history meets risk criteria.",
           (Criterion("Personal or family history meets the plan's risk criteria for the syndrome tested.", "yes"),
            Criterion("Pre-test genetic counseling by a qualified counselor is documented within the prior {v}.", "6 months", "12 months"),
            Criterion("The result will change screening, treatment, or surgical decisions, as documented.", "yes")),
           ("Pedigree or family history form", "Genetic counseling note", "Ordering provider's statement of management impact"),
           ("Direct-to-consumer panels", "Repeat testing of a previously tested gene"),
           date(2025, 2, 1)),
    Policy("CP-0010", "Continuous Glucose Monitoring", "a continuous glucose monitor",
           "Continuous glucose monitoring is covered for members on insulin or with documented problematic hypoglycemia.",
           (Criterion("The member uses insulin, or has had at least {v} documented in the prior six months.", "two hypoglycemic events", "one hypoglycemic event"),
            Criterion("The prescribing provider has seen the member for diabetes management within the prior six months.", "yes"),
            Criterion("Continued coverage requires a visit at least every {v}.", "6 months")),
           ("Diabetes management visit note", "Insulin prescription or hypoglycemia documentation", "Device order"),
           ("Members not on insulin without documented hypoglycemia", "Replacement more often than the manufacturer's stated device life"),
           date(2025, 11, 1)),
    Policy("CP-0011", "Spinal Fusion", "spinal fusion surgery",
           "Spinal fusion is covered for documented instability or deformity after failure of conservative care.",
           (Criterion("Conservative treatment of at least 6 months has failed, documented with dates.", "yes"),
            Criterion("Imaging demonstrates instability, spondylolisthesis, or deformity.", "yes"),
            Criterion("Nicotine use has been discontinued for at least 6 weeks before surgery, as documented.", "yes")),
           ("Conservative treatment history", "Imaging report", "Surgical plan"), ("Fusion for axial back pain without instability",), None),
    Policy("CP-0012", "Varicose Vein Treatment", "varicose vein treatment",
           "Treatment of varicose veins is covered when symptomatic reflux is documented and compression therapy has failed.",
           (Criterion("Duplex ultrasound documents reflux of at least 500 milliseconds.", "yes"),
            Criterion("A trial of compression stockings of at least 3 months has failed, documented.", "yes")),
           ("Duplex ultrasound report", "Compression therapy documentation"), ("Treatment of spider veins", "Cosmetic treatment"), None),
    Policy("CP-0013", "Hyperbaric Oxygen Therapy", "hyperbaric oxygen therapy",
           "Hyperbaric oxygen is covered for listed indications with documented failure of standard wound care where applicable.",
           (Criterion("The diagnosis appears on the plan's list of covered indications.", "yes"),
            Criterion("For diabetic wounds, at least 30 days of standard wound care has failed, documented.", "yes")),
           ("Wound measurements over time", "Treating physician's order"), ("Indications not on the covered list",), None),
    Policy("CP-0014", "Cochlear Implants", "a cochlear implant",
           "Cochlear implants are covered for severe to profound sensorineural hearing loss with limited benefit from hearing aids.",
           (Criterion("Audiometry documents severe to profound bilateral sensorineural hearing loss.", "yes"),
            Criterion("A trial of appropriately fitted hearing aids of at least 3 months shows limited benefit.", "yes")),
           ("Audiogram", "Hearing aid trial documentation"), ("Unilateral loss with normal contralateral hearing",), None),
    Policy("CP-0015", "Growth Hormone Therapy", "growth hormone therapy",
           "Growth hormone is covered for documented deficiency or listed pediatric conditions.",
           (Criterion("Two stimulation tests document deficiency, or a listed condition is documented.", "yes"),
            Criterion("Bone age and growth velocity are documented for pediatric members.", "yes")),
           ("Stimulation test results", "Growth chart"), ("Idiopathic short stature without listed criteria", "Athletic or anti-aging use"), None),
    Policy("CP-0016", "Transcranial Magnetic Stimulation", "transcranial magnetic stimulation",
           "Transcranial magnetic stimulation is covered for major depressive disorder after failure of medication trials.",
           (Criterion("At least two antidepressant trials of adequate dose and duration have failed, documented.", "yes"),
            Criterion("A psychiatrist confirms the diagnosis and the absence of contraindications.", "yes")),
           ("Medication history with dates and doses", "Psychiatric evaluation"), ("Members with implanted metallic devices in the head",), None),
    Policy("CP-0017", "Knee Arthroscopy for Osteoarthritis", "knee arthroscopy",
           "Arthroscopy is covered for mechanical symptoms with imaging correlation; it is not covered for osteoarthritis alone.",
           (Criterion("Mechanical symptoms (locking, catching) are documented.", "yes"),
            Criterion("Imaging shows a lesion that corresponds to the symptoms.", "yes"),
            Criterion("Conservative treatment of at least 12 weeks has failed.", "yes")),
           ("Examination note", "Imaging report", "Conservative treatment history"), ("Lavage or debridement for osteoarthritis alone",), None),
    Policy("CP-0018", "Panniculectomy", "panniculectomy",
           "Panniculectomy is covered when the pannus causes documented recurrent skin conditions unresponsive to treatment.",
           (Criterion("The pannus hangs below the pubis, documented with photographs.", "yes"),
            Criterion("Recurrent intertrigo or cellulitis has persisted for at least 3 months despite treatment.", "yes"),
            Criterion("Weight has been stable for at least 6 months.", "yes")),
           ("Photographs", "Treatment history for skin conditions", "Weight records"), ("Cosmetic abdominoplasty",), None),
    Policy("CP-0019", "Proton Beam Therapy", "proton beam therapy",
           "Proton beam therapy is covered for listed diagnoses where sparing adjacent tissue is clinically necessary.",
           (Criterion("The diagnosis appears on the plan's list of covered indications for proton therapy.", "yes"),
            Criterion("A comparison plan shows a clinically meaningful dose reduction to a critical structure.", "yes")),
           ("Radiation oncology consultation", "Comparative treatment plan"), ("Diagnoses not on the covered list",), None),
    Policy("CP-0020", "Nutritional Counseling and Medical Nutrition Therapy", "medical nutrition therapy",
           "Medical nutrition therapy is covered for diabetes, kidney disease, and other listed conditions on referral.",
           (Criterion("A treating provider's referral documents a listed diagnosis.", "yes"),
            Criterion("Services are provided by a registered dietitian.", "yes"),
            Criterion("Up to 3 hours in the first year and 2 hours in subsequent years are covered without further review.", "yes")),
           ("Referral", "Dietitian's plan"), ("Weight-loss programs without a listed diagnosis",), None),
)


def _render(policy: Policy, version: int, effective_from: date, effective_to: date | None, supersedes: str | None) -> str:
    status = "superseded" if effective_to else "current"
    lines = [
        f"# {policy.pid} {policy.title} (version {version})",
        f"Policy {policy.pid}  version {version}  effective {effective_from.isoformat()}"
        + (f" to {effective_to.isoformat()}  status superseded" if effective_to else "  status current")
        + (f"  supersedes version {version - 1}" if supersedes else ""),
        "",
        "## Purpose",
        policy.purpose,
        "",
        "## Criteria",
        f"Coverage of {policy.service} requires all of the following:",
    ]
    for i, c in enumerate(policy.criteria, start=1):
        value = c.v2 if (version == 2 and c.v2) else c.v1
        lines.append(f"{i}. " + (c.text.format(v=value) if "{v}" in c.text else c.text))
    lines += ["", "## Documentation"] + [f"- {d}" for d in policy.documentation]
    lines += ["", "## Exclusions"] + [f"- {e}" for e in policy.exclusions]
    lines += ["", "## References", f"- GEHA plan brochure, {policy.brochure_ref}.",
              "- Requests that do not meet the criteria are reviewed by a physician reviewer before a denial."]
    if version == 2:
        changed = [c for c in policy.criteria if c.v2]
        lines += ["", "## Revision history",
                  f"- Version 2, effective {effective_from.isoformat()}: " + "; ".join(
                      f"criterion changed from '{c.v1}' to '{c.v2}'" for c in changed) + ".",
                  f"- Version 1, effective {V1_EFFECTIVE.isoformat()}: initial policy."]
    else:
        lines += ["", "## Revision history", f"- Version 1, effective {V1_EFFECTIVE.isoformat()}: initial policy."]
    return "\n".join(lines) + "\n"


def generate(policies_dir: Path = POLICIES_DIR, manifest_path: Path = MANIFEST_PATH) -> dict:
    policies_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for old in policies_dir.glob("CP-*.md"):
        old.unlink()
    docs, versions = 0, 0
    with open(manifest_path, "w") as manifest:
        for policy in POLICIES:
            v1_to = policy.v2_effective
            entries = [(1, V1_EFFECTIVE, v1_to, None)]
            if policy.v2_effective:
                entries.append((2, policy.v2_effective, None, f"{policy.pid}_v1"))
                versions += 1
            for version, eff_from, eff_to, supersedes in entries:
                stem = f"{policy.pid}_v{version}"
                (policies_dir / f"{stem}.md").write_text(_render(policy, version, eff_from, eff_to, supersedes))
                manifest.write(json.dumps({
                    "doc": f"{stem}.md", "template": "policy", "year": eff_from.year,
                    "title": f"{policy.pid} {policy.title} v{version}",
                    "policy_id": policy.pid, "version": version,
                    "effective_from": eff_from.isoformat(), "effective_to": eff_to.isoformat() if eff_to else None,
                    "status": "superseded" if eff_to else "current", "supersedes": supersedes,
                    "entities": [],
                }) + "\n")
                docs += 1
    return {"policies": len(POLICIES), "with_second_version": versions, "documents": docs}
