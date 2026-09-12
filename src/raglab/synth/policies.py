"""Clinical (medical) policies: authored, versioned, deterministic.

CP-0001..CP-0040; a version chain (up to three) per policy, each version a
document with an effective window (Phase 3.5 corpus growth added CP-0021..40
and third versions).

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
    v3: str | None = None  # None: unchanged in version 3

    def value(self, version: int) -> str:
        if version >= 3 and self.v3:
            return self.v3
        if version >= 2 and self.v2:
            return self.v2
        return self.v1


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
    v3_effective: date | None = None  # a third version, after the second

    def effective_dates(self) -> list[date]:
        dates = [V1_EFFECTIVE]
        if self.v2_effective:
            dates.append(self.v2_effective)
            if self.v3_effective:
                dates.append(self.v3_effective)
        return dates


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
            Criterion("The member or caregiver has completed administration training, or nursing visits are arranged for at least the first {v}.", "3 infusions", "5 infusions", "4 infusions"),
            Criterion("A treating physician's order states drug, dose, route, frequency, and duration.", "yes")),
           ("Physician order", "Training attestation or nursing plan", "Pharmacy compounding record when applicable"),
           ("Drugs requiring continuous cardiac monitoring", "First doses of drugs with a documented high anaphylaxis risk"),
           date(2025, 7, 1), v3_effective=date(2026, 2, 1)),
    Policy("CP-0005", "Inpatient Rehabilitation Facility Admission", "inpatient rehabilitation",
           "Inpatient rehabilitation is covered when the member requires intensive multidisciplinary therapy that cannot be provided at a lower level of care.",
           (Criterion("The member can participate in at least {v} of therapy per day, five days per week.", "3 hours", None, "15 hours per week"),
            Criterion("At least two therapy disciplines are required, one of which is physical or occupational therapy.", "yes"),
            Criterion("A physician with rehabilitation training conducts face-to-face visits at least {v}.", "three times per week", "daily")),
           ("Pre-admission screening within 48 hours of admission", "Therapy tolerance assessment", "Physician certification"),
           ("Members who can be safely served by a skilled nursing facility", "Maintenance therapy without measurable goals"),
           date(2025, 3, 1), v3_effective=date(2026, 1, 15)),
    Policy("CP-0006", "Durable Medical Equipment: Power Mobility Devices", "a power wheelchair or scooter",
           "Power mobility devices are covered when a mobility limitation prevents activities of daily living in the home and cannot be resolved with a cane, walker, or manual wheelchair.",
           (Criterion("A face-to-face mobility examination was performed within {v} of the order.", "45 days", "60 days", "90 days"),
            Criterion("The member cannot safely use a manual wheelchair, as documented.", "yes"),
            Criterion("The home permits use of the device, as documented by a home assessment.", "yes")),
           ("Mobility examination note", "Home assessment", "Detailed product description signed by the physician"),
           ("Devices requested primarily for use outside the home", "Upgrades for convenience features"),
           date(2025, 8, 1), v3_effective=date(2026, 3, 1)),
    Policy("CP-0007", "Physical Therapy Visit Limits and Continuation", "continued physical therapy",
           "Physical therapy is covered while measurable functional improvement is documented.",
           (Criterion("An initial evaluation documents functional deficits and measurable goals.", "yes"),
            Criterion("Continuation beyond {v} requires a progress report showing measurable improvement toward the goals.", "12 visits", "20 visits", "24 visits"),
            Criterion("Re-evaluation occurs at least every 30 days.", "yes")),
           ("Initial evaluation with goals", "Progress reports at each continuation point", "Plan of care signed by the treating provider"),
           ("Maintenance programs the member can perform independently", "Duplicate therapy from two providers for the same condition"),
           date(2025, 9, 1), v3_effective=date(2026, 4, 1)),
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
            Criterion("Continued coverage requires a visit at least every {v}.", "6 months", None, "12 months")),
           ("Diabetes management visit note", "Insulin prescription or hypoglycemia documentation", "Device order"),
           ("Members not on insulin without documented hypoglycemia", "Replacement more often than the manufacturer's stated device life"),
           date(2025, 11, 1), v3_effective=date(2026, 2, 15)),
    Policy("CP-0011", "Spinal Fusion", "spinal fusion surgery",
           "Spinal fusion is covered for documented instability or deformity after failure of conservative care.",
           (Criterion("Conservative treatment of at least 6 months has failed, documented with dates.", "yes"),
            Criterion("Imaging demonstrates instability, spondylolisthesis, or deformity.", "yes"),
            Criterion("Nicotine use has been discontinued for at least 6 weeks before surgery, as documented.", "yes")),
           ("Conservative treatment history", "Imaging report", "Surgical plan"), ("Fusion for axial back pain without instability",), None),
    Policy("CP-0012", "Varicose Vein Treatment", "varicose vein treatment",
           "Treatment of varicose veins is covered when symptomatic reflux is documented and compression therapy has failed.",
           (Criterion("Duplex ultrasound documents reflux of at least 500 milliseconds.", "yes"),
            Criterion("A trial of compression stockings of at least {v} has failed, documented.", "3 months", "6 weeks")),
           ("Duplex ultrasound report", "Compression therapy documentation"), ("Treatment of spider veins", "Cosmetic treatment"), date(2025, 4, 15)),
    Policy("CP-0013", "Hyperbaric Oxygen Therapy", "hyperbaric oxygen therapy",
           "Hyperbaric oxygen is covered for listed indications with documented failure of standard wound care where applicable.",
           (Criterion("The diagnosis appears on the plan's list of covered indications.", "yes"),
            Criterion("For diabetic wounds, at least {v} of standard wound care has failed, documented.", "30 days", "60 days")),
           ("Wound measurements over time", "Treating physician's order"), ("Indications not on the covered list",), date(2025, 6, 15)),
    Policy("CP-0014", "Cochlear Implants", "a cochlear implant",
           "Cochlear implants are covered for severe to profound sensorineural hearing loss with limited benefit from hearing aids.",
           (Criterion("Audiometry documents severe to profound bilateral sensorineural hearing loss.", "yes"),
            Criterion("A trial of appropriately fitted hearing aids of at least {v} shows limited benefit.", "3 months", "6 months")),
           ("Audiogram", "Hearing aid trial documentation"), ("Unilateral loss with normal contralateral hearing",), date(2025, 8, 15)),
    Policy("CP-0015", "Growth Hormone Therapy", "growth hormone therapy",
           "Growth hormone is covered for documented deficiency or listed pediatric conditions.",
           (Criterion("Two stimulation tests document deficiency, or a listed condition is documented.", "yes"),
            Criterion("Bone age and growth velocity are documented for pediatric members.", "yes")),
           ("Stimulation test results", "Growth chart"), ("Idiopathic short stature without listed criteria", "Athletic or anti-aging use"), None),
    Policy("CP-0016", "Transcranial Magnetic Stimulation", "transcranial magnetic stimulation",
           "Transcranial magnetic stimulation is covered for major depressive disorder after failure of medication trials.",
           (Criterion("At least {v} antidepressant trials of adequate dose and duration have failed, documented.", "two", "four"),
            Criterion("A psychiatrist confirms the diagnosis and the absence of contraindications.", "yes")),
           ("Medication history with dates and doses", "Psychiatric evaluation"), ("Members with implanted metallic devices in the head",), date(2025, 10, 15)),
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
            Criterion("Weight has been stable for at least {v}.", "6 months", "12 months")),
           ("Photographs", "Treatment history for skin conditions", "Weight records"), ("Cosmetic abdominoplasty",), date(2025, 12, 15)),
    Policy("CP-0019", "Proton Beam Therapy", "proton beam therapy",
           "Proton beam therapy is covered for listed diagnoses where sparing adjacent tissue is clinically necessary.",
           (Criterion("The diagnosis appears on the plan's list of covered indications for proton therapy.", "yes"),
            Criterion("A comparison plan shows a clinically meaningful dose reduction to a critical structure.", "yes")),
           ("Radiation oncology consultation", "Comparative treatment plan"), ("Diagnoses not on the covered list",), None),
    Policy("CP-0020", "Nutritional Counseling and Medical Nutrition Therapy", "medical nutrition therapy",
           "Medical nutrition therapy is covered for diabetes, kidney disease, and other listed conditions on referral.",
           (Criterion("A treating provider's referral documents a listed diagnosis.", "yes"),
            Criterion("Services are provided by a registered dietitian.", "yes"),
            Criterion("Up to {v} in the first year and 2 hours in subsequent years are covered without further review.", "3 hours", "4 hours")),
           ("Referral", "Dietitian's plan"), ("Weight-loss programs without a listed diagnosis",), date(2026, 1, 15)),
    # ---- CP-0021..CP-0040 (Phase 3.5 corpus growth): sibling topics of the
    # policies above, so the library crowds the way a real one does.
    Policy("CP-0021", "Advanced Imaging: PET and PET/CT", "PET or PET/CT imaging",
           "PET imaging is covered for listed oncologic, cardiac, and neurologic indications when the result will change management.",
           (Criterion("The indication appears on the plan's list of covered PET indications.", "yes"),
            Criterion("For oncologic staging, no PET has been performed within the prior {v}.", "90 days", "60 days", "45 days"),
            Criterion("The ordering provider documents the management decision the study will inform.", "yes")),
           ("Ordering provider's note", "Pathology or prior imaging report", "Treatment plan when restaging"),
           ("Screening in the absence of a diagnosis", "Surveillance imaging without symptoms or findings"),
           date(2024, 9, 1), v3_effective=date(2026, 2, 1)),
    Policy("CP-0022", "Occupational Therapy Visit Limits and Continuation", "continued occupational therapy",
           "Occupational therapy is covered while measurable improvement in activities of daily living is documented.",
           (Criterion("An initial evaluation documents deficits in activities of daily living and measurable goals.", "yes"),
            Criterion("Continuation beyond {v} requires a progress report showing measurable improvement.", "12 visits", "16 visits"),
            Criterion("Re-evaluation occurs at least every 30 days.", "yes")),
           ("Initial evaluation with goals", "Progress reports at each continuation point", "Plan of care"),
           ("Maintenance programs the member can perform independently", "Work-hardening programs"),
           date(2025, 9, 1)),
    Policy("CP-0023", "Speech-Language Therapy", "speech-language therapy",
           "Speech-language therapy is covered for a documented disorder resulting from illness, injury, or a developmental condition.",
           (Criterion("A physician or qualified provider documents a diagnosis causing the speech, language, or swallowing disorder.", "yes"),
            Criterion("Continuation beyond {v} requires a progress report with standardized measures.", "20 visits", "24 visits", "30 visits"),
            Criterion("Treatment is delivered by a licensed speech-language pathologist.", "yes")),
           ("Evaluation with standardized measures", "Progress reports", "Referral or order"),
           ("Therapy for accent modification or voice training", "Services delivered by a school as part of an education plan"),
           date(2025, 6, 1), v3_effective=date(2026, 4, 1)),
    Policy("CP-0024", "Skilled Nursing Facility Admission", "a skilled nursing facility admission",
           "Skilled nursing facility care is covered when daily skilled nursing or therapy is required after a qualifying hospital stay.",
           (Criterion("A hospital stay of at least {v} preceded the admission within the prior 30 days.", "3 days", "2 days"),
            Criterion("The member requires skilled nursing or therapy services daily that can only be delivered in a facility.", "yes"),
            Criterion("A physician certifies the need at admission and every 30 days.", "yes")),
           ("Hospital discharge summary", "Skilled need assessment", "Physician certification"),
           ("Custodial care", "Stays beyond the brochure's day limit"),
           date(2025, 2, 1)),
    Policy("CP-0025", "Home Health Skilled Nursing", "home health skilled nursing",
           "Home health skilled nursing is covered for a homebound member who needs intermittent skilled care under a physician's plan.",
           (Criterion("The member is homebound, as documented by the treating physician.", "yes"),
            Criterion("A face-to-face encounter occurred within the {v} before the start of care.", "90 days", "60 days", "30 days"),
            Criterion("The plan of care is reviewed and re-certified at least every 60 days.", "yes")),
           ("Plan of care", "Face-to-face encounter note", "Homebound documentation"),
           ("Custodial or companion services", "Care that a caregiver has been trained to provide"),
           date(2024, 11, 1), v3_effective=date(2026, 1, 15)),
    Policy("CP-0026", "Durable Medical Equipment: Hospital Beds and Support Surfaces", "a hospital bed or support surface",
           "Hospital beds and pressure-reducing support surfaces are covered when the member's condition requires positioning that an ordinary bed cannot provide.",
           (Criterion("The medical record documents a condition requiring positioning, such as severe cardiopulmonary disease or a stage {v} pressure injury.", "3", "2"),
            Criterion("The prescribing provider has seen the member within the prior 6 months.", "yes"),
            Criterion("For a support surface, a documented pressure injury or high risk with a validated scale is present.", "yes")),
           ("Prescribing provider's note", "Detailed product description", "Wound assessment when applicable"),
           ("Beds requested for caregiver convenience", "Upgrades for comfort features"),
           date(2025, 8, 1)),
    Policy("CP-0027", "Bariatric Surgery Revision", "bariatric surgery revision",
           "Revision of a prior bariatric procedure is covered for a documented complication or documented failure to achieve expected weight loss despite adherence.",
           (Criterion("A complication of the original procedure is documented, or weight loss is less than {v} of excess weight at 18 months despite documented adherence.", "50%", "40%", "30%"),
            Criterion("The original procedure was medically indicated and performed at least 18 months earlier.", "yes"),
            Criterion("Nutritional and psychological evaluations within the prior 12 months find no contraindication.", "yes")),
           ("Operative report of the original procedure", "Weight records", "Adherence documentation"),
           ("Revision for cosmetic reasons", "Conversion for weight regain without documented adherence"),
           date(2025, 10, 1), v3_effective=date(2026, 3, 1)),
    Policy("CP-0028", "Pharmacogenomic Testing", "pharmacogenomic testing",
           "Pharmacogenomic testing is covered for listed drug-gene pairs when the result will guide prescribing.",
           (Criterion("The drug-gene pair appears on the plan's covered list.", "yes"),
            Criterion("The prescribing provider documents that the result will change drug or dose selection.", "yes")),
           ("Prescribing provider's note", "Test order naming the gene and drug"),
           ("Multi-gene panels beyond the listed pairs", "Repeat testing of a previously tested gene"), None),
    Policy("CP-0029", "Insulin Pumps", "an insulin pump",
           "Insulin pumps are covered for members on multiple daily injections with documented need for pump therapy.",
           (Criterion("The member has been on multiple daily injections for at least {v} with documented adherence.", "6 months", "3 months"),
            Criterion("Glycemic control is inadequate or hypoglycemia is documented despite adherence.", "yes"),
            Criterion("The member has completed pump training with a diabetes educator.", "yes")),
           ("Diabetes management visit notes", "Glucose logs or monitor download", "Training attestation"),
           ("Replacement within the manufacturer's warranty", "Members not on insulin"),
           date(2025, 11, 1)),
    Policy("CP-0030", "Cervical Spinal Fusion", "cervical spinal fusion",
           "Cervical fusion is covered for documented instability, myelopathy, or radiculopathy after failure of conservative care.",
           (Criterion("Conservative treatment of at least {v} has failed, documented with dates, unless myelopathy is present.", "6 weeks", "12 weeks", "8 weeks"),
            Criterion("Imaging demonstrates a lesion that corresponds to the symptoms.", "yes"),
            Criterion("Nicotine use has been discontinued for at least 6 weeks before surgery, as documented.", "yes")),
           ("Conservative treatment history", "Imaging report", "Surgical plan"),
           ("Fusion for axial neck pain without instability or neurologic findings",),
           date(2025, 4, 1), v3_effective=date(2026, 5, 1)),
    Policy("CP-0031", "Lumbar Epidural Steroid Injections", "a lumbar epidural steroid injection",
           "Epidural steroid injections are covered for radicular pain after conservative care, with a limit on the number per year.",
           (Criterion("Radicular pain has persisted for at least {v} despite conservative treatment.", "4 weeks", "6 weeks"),
            Criterion("Imaging or electrodiagnostic findings correlate with the symptoms.", "yes"),
            Criterion("No more than 4 injections per region per 12 months; a repeat injection requires documented relief of at least 50% for 6 weeks.", "yes")),
           ("Examination note", "Imaging report", "Response to prior injections"),
           ("Injections for axial back pain without radicular findings", "Series of injections scheduled without assessing response"),
           date(2025, 3, 1)),
    Policy("CP-0032", "Radiofrequency Ablation for Facet Joint Pain", "radiofrequency ablation of a facet joint",
           "Radiofrequency ablation is covered after positive diagnostic medial branch blocks.",
           (Criterion("Two diagnostic medial branch blocks each produced at least 80% relief for the duration of the anesthetic.", "yes"),
            Criterion("Pain has persisted for at least 3 months despite conservative treatment.", "yes")),
           ("Block procedure notes with relief scores", "Conservative treatment history"),
           ("Repeat ablation within 6 months of a prior ablation at the same level",), None),
    Policy("CP-0033", "Cardiac Rehabilitation", "cardiac rehabilitation",
           "Cardiac rehabilitation is covered after a qualifying cardiac event or procedure, for a limited number of sessions.",
           (Criterion("A qualifying event or procedure occurred within the prior {v}.", "12 months", "6 months", "9 months"),
            Criterion("A physician refers the member and supervises the program.", "yes"),
            Criterion("Up to 36 sessions; continuation beyond 36 requires documented functional gain.", "yes")),
           ("Referral naming the qualifying event", "Program plan", "Progress notes"),
           ("Maintenance programs after completion", "Programs without physician supervision"),
           date(2025, 7, 1), v3_effective=date(2026, 2, 15)),
    Policy("CP-0034", "Positive Airway Pressure Devices (CPAP and BiPAP)", "a CPAP or BiPAP device",
           "Positive airway pressure devices are covered for documented sleep apnea, with continued coverage tied to adherence.",
           (Criterion("A sleep study documents an apnea-hypopnea index of at least 15, or at least 5 with documented symptoms.", "yes"),
            Criterion("Continued coverage after the {v} trial requires adherence of at least 4 hours per night on 70% of nights.", "90-day", "60-day", "90-day"),
            Criterion("A face-to-face visit documents benefit during the trial.", "yes")),
           ("Sleep study report", "Adherence download", "Follow-up visit note"),
           ("Devices for snoring without documented apnea", "Replacement within the device's expected life"),
           date(2025, 5, 1), v3_effective=date(2026, 3, 15)),
    Policy("CP-0035", "Oral Appliances for Obstructive Sleep Apnea", "an oral appliance for sleep apnea",
           "Custom oral appliances are covered for mild to moderate sleep apnea when positive airway pressure is not tolerated or is declined.",
           (Criterion("A sleep study documents mild to moderate obstructive sleep apnea.", "yes"),
            Criterion("The member has failed or declined positive airway pressure therapy, documented.", "yes"),
            Criterion("The appliance is custom fabricated by a licensed dentist.", "yes")),
           ("Sleep study report", "Documentation of PAP intolerance or refusal"),
           ("Over-the-counter appliances", "Appliances for snoring alone"), None),
    Policy("CP-0036", "Botulinum Toxin for Chronic Migraine", "botulinum toxin for chronic migraine",
           "Botulinum toxin is covered for chronic migraine after failure of preventive medications.",
           (Criterion("Headache occurs on at least 15 days per month for at least 3 months, with migraine features on at least 8 days.", "yes"),
            Criterion("At least {v} preventive medication classes have failed after adequate trials.", "two", "three"),
            Criterion("Continuation requires a documented reduction in headache days after two treatment cycles.", "yes")),
           ("Headache diary", "Preventive medication history", "Treatment response documentation"),
           ("Episodic migraine", "Tension-type headache"),
           date(2025, 12, 1)),
    Policy("CP-0037", "Biologic Therapy for Plaque Psoriasis", "biologic therapy for plaque psoriasis",
           "Biologic therapy is covered for moderate to severe plaque psoriasis after failure of conventional systemic therapy.",
           (Criterion("Body surface area involvement of at least 10%, or involvement of the hands, feet, face, or genitals.", "yes"),
            Criterion("Failure, intolerance, or contraindication to at least {v} of conventional systemic therapy or phototherapy.", "3 months", "8 weeks", "12 weeks"),
            Criterion("The prescriber is a dermatologist or rheumatologist.", "yes")),
           ("Dermatology note with body surface area", "Prior therapy history", "Tuberculosis screening"),
           ("Concurrent use of two biologics", "Mild psoriasis"),
           date(2025, 1, 15), v3_effective=date(2026, 1, 1)),
    Policy("CP-0038", "GLP-1 Receptor Agonists for Weight Management", "a GLP-1 receptor agonist for weight management",
           "GLP-1 receptor agonists prescribed for weight management are covered for members who meet body mass index thresholds after a supervised program.",
           (Criterion("Body mass index of at least 30, or at least 27 with an obesity-related comorbidity.", "yes"),
            Criterion("Completion of a physician-supervised weight management program of at least {v}.", "6 months", None, "3 months"),
            Criterion("Continuation beyond 6 months requires a documented weight loss of at least {v} from baseline.", "5%", "4%")),
           ("Weight management program records with dates", "Body mass index and comorbidity documentation", "Prescriber's plan"),
           ("Prescriptions for type 2 diabetes (see the formulary; no review under this policy)", "Concurrent use of two GLP-1 agents"),
           date(2025, 6, 15), v3_effective=date(2026, 1, 1)),
    Policy("CP-0039", "Hearing Aids", "hearing aids",
           "Hearing aids are covered for documented hearing loss, with a replacement interval.",
           (Criterion("Audiometry within the prior 12 months documents a hearing loss of at least 30 decibels in the affected ear.", "yes"),
            Criterion("The device is dispensed by a licensed audiologist or hearing aid dispenser.", "yes"),
            Criterion("Replacement is limited to once every 3 years unless the device is lost or the loss has changed.", "yes")),
           ("Audiogram", "Dispensing record"),
           ("Over-the-counter amplifiers", "Accessories not needed for the device to function"), None),
    Policy("CP-0040", "Negative Pressure Wound Therapy", "negative pressure wound therapy",
           "Negative pressure wound therapy is covered for listed wound types after documented failure of standard wound care.",
           (Criterion("The wound is a listed type (diabetic ulcer, pressure injury stage 3 or 4, surgical dehiscence, or traumatic wound).", "yes"),
            Criterion("At least {v} of standard wound care has failed, documented with wound measurements.", "30 days", "14 days"),
            Criterion("Continuation beyond 4 months requires documented wound size reduction.", "yes")),
           ("Wound measurements over time", "Treating physician's order", "Standard care history"),
           ("Wounds with untreated osteomyelitis", "Wounds with exposed vessels or organs"),
           date(2025, 9, 15)),
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
        value = c.value(version)
        lines.append(f"{i}. " + (c.text.format(v=value) if "{v}" in c.text else c.text))
    lines += ["", "## Documentation"] + [f"- {d}" for d in policy.documentation]
    lines += ["", "## Exclusions"] + [f"- {e}" for e in policy.exclusions]
    lines += ["", "## References", f"- GEHA plan brochure, {policy.brochure_ref}.",
              "- Requests that do not meet the criteria are reviewed by a physician reviewer before a denial."]
    lines += ["", "## Revision history"]
    dates = policy.effective_dates()
    for k in range(version, 1, -1):
        changed = [c for c in policy.criteria if c.value(k) != c.value(k - 1)]
        lines.append(f"- Version {k}, effective {dates[k - 1].isoformat()}: " + "; ".join(
            f"criterion changed from '{c.value(k - 1)}' to '{c.value(k)}'" for c in changed) + ".")
    lines.append(f"- Version 1, effective {V1_EFFECTIVE.isoformat()}: initial policy.")
    return "\n".join(lines) + "\n"


def generate(policies_dir: Path = POLICIES_DIR, manifest_path: Path = MANIFEST_PATH) -> dict:
    policies_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for old in policies_dir.glob("CP-*.md"):
        old.unlink()
    docs, versions, thirds = 0, 0, 0
    with open(manifest_path, "w") as manifest:
        for policy in POLICIES:
            dates = policy.effective_dates()
            entries = []
            for i, eff_from in enumerate(dates, start=1):
                eff_to = dates[i] if i < len(dates) else None
                entries.append((i, eff_from, eff_to, f"{policy.pid}_v{i - 1}" if i > 1 else None))
            versions += len(dates) >= 2
            thirds += len(dates) >= 3
            previous = None
            for version, eff_from, eff_to, supersedes in entries:
                stem = f"{policy.pid}_v{version}"
                text = _render(policy, version, eff_from, eff_to, supersedes)
                if previous is not None and not any(c.value(version) != c.value(version - 1) for c in policy.criteria):
                    raise ValueError(f"{policy.pid} version {version} changes no criterion")
                (policies_dir / f"{stem}.md").write_text(text)
                previous = text
                manifest.write(json.dumps({
                    "doc": f"{stem}.md", "template": "policy", "year": eff_from.year,
                    "title": f"{policy.pid} {policy.title} v{version}",
                    "policy_id": policy.pid, "version": version,
                    "effective_from": eff_from.isoformat(), "effective_to": eff_to.isoformat() if eff_to else None,
                    "status": "superseded" if eff_to else "current", "supersedes": supersedes,
                    "entities": [],
                }) + "\n")
                docs += 1
    return {"policies": len(POLICIES), "with_second_version": versions, "with_third_version": thirds, "documents": docs}
