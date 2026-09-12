"""The internal library grown to a realistic size (Phase 3.5): bulletin
series across three years with supersession, SOPs with archived prior
versions, knowledge-base articles per plan family, and one formulary per
year with quarterly update logs.

Everything here is synthetic. Real documents (brochures, carrier letters,
the Synthea-derived population) are never grown: the corpus should be as
hard as reality, not harder. Each document carries the year of its content
and a record (issued / effective window / supersession) that the ingest
writes into chunk metadata, so version precedence applies to these sources
the way it applies to clinical policies.
"""

from raglab.synth.internal_docs import InternalDoc


def _bulletin(number: str, title: str, issued: str, audience: str, body: str,
              supersedes: str | None = None, superseded_by: str | None = None,
              effective_to: str | None = None) -> InternalDoc:
    year = int(number[:4])
    head = f"# Claims Bulletin {number} — {title}\n\n**Issued:** {issued} · **Audience:** {audience}\n"
    if supersedes:
        head += f"\nSupersedes Bulletin {supersedes} on this subject.\n"
    if superseded_by:
        head += f"\n*Superseded by Bulletin {superseded_by}; retained for claims with earlier service dates.*\n"
    record = {"bulletin_id": number, "effective_from": issued, "effective_to": effective_to,
              "status": "superseded" if effective_to else "current", "supersedes": supersedes}
    return InternalDoc(f"bulletins/bulletin_{number.replace('-', '_')}.md", "employee", "bulletin",
                       f"Claims Bulletin {number}: {title}", head + "\n" + body.strip() + "\n",
                       year=year, record=record)


BULLETIN_SERIES = [
    # ---- 2024 ---------------------------------------------------------
    _bulletin("2024-001", "Timely Filing Limits", "2024-01-08", "all examiners", """
- Claims must be received within 365 days of the date of service. Claims
  received later deny with reason code TF-01 unless the provider documents
  a qualifying delay (retroactive eligibility, coordination with a primary
  payer, or a Plan processing error).
- A corrected claim inherits the original claim's received date.
- Do not apply the limit to claims where the Plan is secondary and the
  primary payer's explanation of benefits is dated within the last 90 days.
""", superseded_by="2025-003", effective_to="2025-01-13"),
    _bulletin("2024-002", "Telehealth Cost Share", "2024-02-05", "all examiners", """
- Telehealth visits with an in-network provider adjudicate at the office
  visit cost share of the same specialty for the member's option.
- Audio-only visits are covered when the provider documents that video was
  not available to the member; adjudicate at the same cost share.
- Out-of-network telehealth: apply the out-of-network office visit share.
""", superseded_by="2025-004", effective_to="2025-02-03"),
    _bulletin("2024-003", "Durable Medical Equipment Rental to Purchase", "2024-03-04", "DME unit, all examiners", """
- Rental of durable medical equipment converts to purchase after 10 months
  of continuous rental; the Plan's total payments cap at the purchase
  allowance.
- Oxygen equipment rents for 36 months and does not convert.
- Repairs to member-owned equipment are covered after the manufacturer
  warranty; deny repairs within the warranty period with a note to the
  supplier.
""", superseded_by="2026-007", effective_to="2026-03-02"),
    _bulletin("2024-004", "Ambulance Billing", "2024-05-06", "all examiners", """
- Ground ambulance is covered when transport by any other means would
  endanger the member's health; apply the emergency cost share.
- Air ambulance requires medical necessity review unless the transport
  originated from an emergency room with a documented level-of-care need.
- Mileage is payable to the nearest appropriate facility only; excess
  mileage denies with reason code AM-03.
"""),
    _bulletin("2024-005", "Preventive Service Coding", "2024-07-01", "all examiners", """
- In-network services billed with a preventive modifier and a preventive
  diagnosis adjudicate at no member cost share and bypass the deductible.
- A diagnostic service performed during a preventive visit (for example a
  polyp removal during a screening colonoscopy) remains preventive when the
  screening indication is documented.
- Out-of-network preventive services apply the standard out-of-network
  cost share.
""", superseded_by="2025-006", effective_to="2025-07-07"),
    _bulletin("2024-006", "Coordination of Benefits: Order of Payment", "2024-09-09", "COB unit", """
- When two group health plans cover a member, the plan covering the member
  as an employee pays before the plan covering the member as a dependent.
- For dependent children, the parent whose birthday falls earlier in the
  calendar year is primary (birthday rule); a court order overrides it.
- Verify other coverage on the eligibility file before paying secondary.
"""),
    # ---- 2025 ---------------------------------------------------------
    _bulletin("2025-001", "Postal Service Health Benefits Launch", "2025-01-06", "intake, all examiners", """
- Postal Service employees and annuitants are enrolled in PSHB contracts
  from 2025-01-01: 71-021 (High/Standard), 71-022 (Elevate/Elevate Plus),
  71-026 (HDHP). Verify the program before applying benefit rules.
- FEHB and PSHB accumulators are separate even when the option names match.
- Claims with 2024 service dates for members who moved to PSHB adjudicate
  under the member's 2024 FEHB enrollment.
""", superseded_by="2026-003", effective_to="2026-01-20"),
    _bulletin("2025-002", "Medicare Part B Premium Reimbursement", "2025-01-13", "member services, COB unit", """
- Annuitants enrolled in Medicare Part B under a High Option enrollment are
  eligible for a Part B premium reimbursement account; the reimbursement is
  processed by the Plan's account administrator, not through claims.
- Income-related monthly adjustment amounts (IRMAA) are not reimbursed.
  Direct members who ask about IRMAA to the Social Security Administration.
- Reimbursement requests need proof of premium payment for the period.
""", superseded_by="2026-008", effective_to="2026-04-06"),
    _bulletin("2025-003", "Timely Filing Limits", "2025-01-13", "all examiners", """
- Claims must be received within 365 days of the date of service (unchanged).
- New: claims from a provider who was retroactively credentialed are
  accepted within 180 days of the credentialing date; document reason code
  TF-04.
- A corrected claim inherits the original claim's received date.
""", supersedes="2024-001", superseded_by="2026-005", effective_to="2026-01-26"),
    _bulletin("2025-004", "Telehealth Cost Share", "2025-02-03", "all examiners", """
- Telehealth visits with an in-network provider adjudicate at the office
  visit cost share of the same specialty for the member's option.
- Audio-only visits no longer require documentation that video was
  unavailable; adjudicate at the same cost share as video.
- Telehealth through the Plan's designated telehealth vendor is covered at
  no member cost share under every option.
""", supersedes="2024-002"),
    _bulletin("2025-005", "No Surprises Act: Out-of-Network Emergency Services", "2025-03-03", "all examiners", """
- Emergency services from an out-of-network facility or provider adjudicate
  at the in-network cost share; the member cannot be balance billed.
- Post-stabilization services at an out-of-network facility remain protected
  until the member can be safely transferred and has consented in writing.
- Air ambulance claims are protected; ground ambulance claims are not.
"""),
    _bulletin("2025-006", "Preventive Service Coding", "2025-07-07", "all examiners", """
- In-network preventive services adjudicate at no member cost share and
  bypass the deductible (unchanged).
- New: breast cancer screening is preventive from age 40 (previously 50)
  for service dates on or after 2025-07-01.
- A diagnostic service performed during a preventive visit remains
  preventive when the screening indication is documented.
""", supersedes="2024-005"),
    _bulletin("2025-007", "GLP-1 Medications: Prior Authorization", "2025-09-08", "pharmacy, member services", """
- GLP-1 receptor agonists prescribed for weight management require prior
  authorization under medical policy CP-0038; prescriptions for type 2
  diabetes adjudicate under the formulary tier without CP-0038 review.
- Pharmacy claims rejected for missing authorization return message PA-11;
  advise members that the prescriber must submit the request.
- Quantity limits: a 30-day supply at retail, 90 days at mail order.
""", superseded_by="2026-009", effective_to="2026-05-04"),
    _bulletin("2025-008", "Chiropractic and Acupuncture Visit Limits", "2025-11-03", "all examiners", """
- Chiropractic manipulation and acupuncture visits count against the
  annual visit limit stated in the member's brochure for the plan year of
  the service date.
- Visits beyond the limit deny with reason code VL-02; the limit resets
  January 1.
- Evaluation and management billed on the same day as manipulation by the
  same provider is bundled; do not pay separately.
"""),
    # ---- 2026 (2026-001..004 are authored in internal_docs.py) ---------
    _bulletin("2026-005", "Timely Filing Limits", "2026-01-26", "all examiners", """
- Claims must be received within 365 days of the date of service.
- Retroactively credentialed providers: 180 days from the credentialing
  date (reason code TF-04).
- New: claims where the Plan is secondary must be received within 90 days
  of the primary payer's explanation of benefits date or within the 365-day
  limit, whichever is later.
""", supersedes="2025-003"),
    _bulletin("2026-006", "Urgent Care Center Cost Share", "2026-02-09", "all examiners", """
- Urgent care visits adjudicate at the urgent care copay stated in the
  member's brochure for the plan year; they are not emergency room visits
  even when the center is hospital-owned.
- A facility billing an urgent care visit with an emergency room revenue
  code is re-coded to urgent care when the place of service is urgent care.
- Telehealth urgent care through the designated vendor remains no cost
  share (see Bulletin 2025-004).
"""),
    _bulletin("2026-007", "Durable Medical Equipment Rental to Purchase", "2026-03-02", "DME unit, all examiners", """
- Rental converts to purchase after 13 months of continuous rental (was 10
  months). Apply the new term to rentals beginning on or after 2026-03-01;
  rentals already in progress keep the 10-month term.
- Oxygen equipment rents for 36 months and does not convert (unchanged).
- Continuous positive airway pressure devices follow the 13-month term and
  require the adherence documentation in medical policy CP-0034.
""", supersedes="2024-003"),
    _bulletin("2026-008", "Medicare Part B Premium Reimbursement", "2026-04-06", "member services, COB unit", """
- The Part B premium reimbursement account is available to annuitants
  under High Option and, new for 2026, under Standard Option enrollments
  with Medicare Part A and B.
- Income-related monthly adjustment amounts (IRMAA) are not reimbursed;
  the brochure Section 9 states the base premium amount only. Direct IRMAA
  questions to the Social Security Administration.
- Reimbursement requests need proof of premium payment for the period.
""", supersedes="2025-002"),
    _bulletin("2026-009", "GLP-1 Medications: Prior Authorization", "2026-05-04", "pharmacy, member services", """
- GLP-1 receptor agonists for weight management require prior authorization
  under medical policy CP-0038 version 3 (effective 2026-01-01): the
  physician-supervised program is now 3 months (was 6).
- Prescriptions for type 2 diabetes adjudicate under the formulary tier
  without CP-0038 review; step therapy through metformin applies to new
  starts (Formulary 2026).
- Quantity limits unchanged: 30 days at retail, 90 days at mail order.
""", supersedes="2025-007"),
    _bulletin("2026-010", "Overpayment Recovery", "2026-06-01", "recovery unit, all examiners", """
- Provider overpayments identified within 12 months of payment are offset
  against future claims after a 30-day written notice.
- Member overpayments are recovered by invoice; do not offset against a
  member's other claims.
- Overpayments older than 24 months are written off unless fraud is
  suspected, in which case route to the special investigations unit.
"""),
]


def _sop(stem: str, title: str, applies_to: str, effective: str, body: str,
         version: int = 1, effective_to: str | None = None, supersedes: str | None = None,
         archived: bool = False) -> InternalDoc:
    year = int(effective[:4])
    head = (f"# {title}\n\n**Applies to:** {applies_to}\n**Effective:** {effective}"
            + (f" · **Version:** {version}" if version > 1 or archived else "")
            + (f" · supersedes version {version - 1}" if supersedes else "") + "\n")
    if archived:
        head += f"\n*Archived version; superseded on {effective_to}. Retained for audits of claims processed under it.*\n"
    relpath = f"sops/{stem}{'_v' + str(version) if archived else ''}.md"
    record = {"sop_id": stem, "version": version, "effective_from": effective, "effective_to": effective_to,
              "status": "superseded" if effective_to else "current", "supersedes": supersedes}
    return InternalDoc(relpath, "employee", "sop", title + (f" (v{version}, archived)" if archived else ""),
                       head + "\n" + body.strip() + "\n", year=year, record=record)


# Archived 2025 versions of the four SOPs authored in internal_docs.py: the
# current versions there are version 2, effective 2026-01-01.
SOP_ARCHIVE = [
    _sop("sop_deductible_verification", "SOP: HDHP Deductible Verification", "Claims examiners, tier 1-2", "2025-01-01", """
## Procedure

1. Confirm member enrollment option. For HDHP (71-014) the 2025 in-network
   calendar year deductible is $1,600 Self Only / $3,200 Self Plus One or
   Self and Family. Out-of-network: $4,000 / $8,000.
2. Preventive care services bypass the deductible entirely when rendered
   in-network.
3. Deductible accumulators reset January 1. Claims spanning the year boundary
   apply to the year of the service date.
4. If the member changed enrollment options mid-year, credit covered expenses
   already applied under the prior option toward the new option's deductible.
""", version=1, effective_to="2026-01-01", archived=True),
    _sop("sop_prior_authorization", "SOP: Prior Authorization Intake", "Utilization review intake", "2025-01-01", """
## Procedure

1. Verify the requesting provider is identified and the member ID is valid.
2. Non-urgent care requests: decision within 30 calendar days.
3. Urgent care claims: decision within 72 hours of receipt.
4. Emergency inpatient admissions do not require pre-approval; notification
   is required within 2 business days following the admission.
5. Missing clinical documentation: deny on first pass with a request for
   the documentation; the provider may resubmit.
""", version=1, effective_to="2026-01-01", archived=True),
    _sop("sop_claims_escalation", "SOP: Claims Escalation Thresholds", "Claims examiners", "2025-01-01", """
## Escalation rules

- Claims with billed amount over $25,000: senior examiner review required
  before adjudication.
- Claims involving out-of-network emergency services: verify No Surprises Act
  balance-billing protections before applying member cost share.
- Potential coordination-of-benefits: route to the COB unit.
- Suspected fraud indicators: route directly to SIU.
""", version=1, effective_to="2026-01-01", archived=True),
    _sop("sop_cob_medicare", "SOP: Coordination of Benefits with Medicare", "COB unit", "2025-01-01", """
## Order of benefits

1. For active federal employees age 65+, the Plan pays primary and Medicare
   pays secondary.
2. For annuitants enrolled in Medicare Part A and B, Medicare pays primary
   and the Plan pays secondary.
3. Verify Medicare enrollment status via the eligibility file before
   adjudicating any claim flagged age 65 or over.
""", version=1, effective_to="2026-01-01", archived=True),
]

SOP_SERIES = [
    _sop("sop_timely_filing_review", "SOP: Timely Filing Review", "Claims examiners", "2026-01-26", """
## Procedure

1. Compare the received date with the date of service; the limit is 365
   days (Bulletin 2026-005).
2. For denials with reason code TF-01, check for a qualifying delay before
   finalizing: retroactive eligibility, a primary payer's explanation of
   benefits within 90 days, or a Plan processing error.
3. Retroactively credentialed providers: accept within 180 days of the
   credentialing date and document TF-04.
4. Log every override with the qualifying reason; overrides without a
   reason are reversed at audit.
"""),
    _sop("sop_duplicate_claims", "SOP: Duplicate Claim Detection", "Claims examiners", "2026-01-01", """
## Procedure

1. A claim with the same member, provider, date of service, and procedure
   code as a paid claim is a suspected duplicate.
2. Check for a corrected-claim indicator (frequency code 7 or 8) before
   denying; a corrected claim replaces the original and is not a duplicate.
3. Bilateral procedures and multiple units on one line are not duplicates
   when the modifier documents them.
4. Deny confirmed duplicates with reason code DUP-01 and reference the
   paid claim number in the remittance.
"""),
    _sop("sop_provider_disputes", "SOP: Provider Payment Disputes", "Provider relations, senior examiners", "2026-01-01", """
## Procedure

1. A provider dispute is accepted within 180 days of the remittance date.
2. Pricing disputes on in-network claims are checked against the contracted
   fee schedule in effect on the date of service.
3. Out-of-network emergency disputes follow the No Surprises Act
   independent dispute resolution timeline (Bulletin 2025-005).
4. Respond in writing within 45 days; an unresolved dispute after 60 days
   escalates to the provider relations manager.
"""),
    _sop("sop_member_appeal_intake", "SOP: Member Appeal Intake (First Level)", "Appeals unit", "2026-01-01", """
## Procedure

1. A member may appeal a claim denial within 180 days of the denial date.
2. Log the appeal with the claim number, the denial reason, and the medical
   policy applied, if any (the policy version in effect on the date of
   service governs the review).
3. Clinical appeals are reviewed by a physician reviewer not involved in
   the original decision; administrative appeals by a senior examiner.
4. Decide within 30 days for post-service appeals and 72 hours for urgent
   pre-service appeals; send the determination letter with the OPM review
   rights notice.
"""),
    _sop("sop_oon_emergency_pricing", "SOP: Out-of-Network Emergency Pricing", "Claims examiners", "2026-01-01", """
## Procedure

1. Confirm the claim is for emergency services or protected post-stabilization
   care (Bulletin 2025-005).
2. Price the claim at the qualifying payment amount for the service and
   geographic area; apply the in-network cost share.
3. Suppress balance billing on the remittance and the member's explanation
   of benefits.
4. Route provider objections to the dispute process; do not reprice
   manually.
"""),
    _sop("sop_overpayment_recovery", "SOP: Overpayment Recovery", "Recovery unit", "2026-06-01", """
## Procedure

1. Confirm the overpayment amount and cause before any notice.
2. Provider overpayments: send a 30-day written notice; offset against
   future claims after the notice period (Bulletin 2026-010).
3. Member overpayments: invoice; never offset against a member's claims.
4. Suspected fraud: stop recovery and route to the special investigations
   unit.
"""),
    _sop("sop_enrollment_reconciliation", "SOP: Enrollment Reconciliation", "Enrollment unit, intake", "2026-01-01", """
## Procedure

1. A claim whose member is not found on the eligibility file for the date
   of service pends for 10 business days while enrollment is verified with
   the employing office or OPM.
2. FEHB and PSHB enrollments are reconciled separately; a member found under
   the other program is rerouted, not denied (Bulletin 2026-003).
3. Retroactive enrollment changes reprocess affected claims automatically;
   do not adjust manually.
"""),
    _sop("sop_pharmacy_prior_auth", "SOP: Pharmacy Prior Authorization Intake", "Pharmacy services", "2026-01-01", """
## Procedure

1. Prescriber submits the request with the diagnosis and prior therapy.
2. Check the formulary for the plan year: tier, prior authorization flag,
   step therapy, and quantity limit.
3. Weight-management GLP-1 requests are reviewed under medical policy
   CP-0038; diabetes GLP-1 requests are reviewed against step therapy only
   (Bulletin 2026-009).
4. Decide within 72 hours (24 hours when marked urgent); a denial states the
   criterion not met and the appeal rights.
"""),
]


def _kb(stem: str, title: str, body: str) -> InternalDoc:
    return InternalDoc(f"kb/{stem}.md", "employee", "kb", f"CSR KB: {title}",
                       f"# CSR KB: {title} (2026)\n\n" + body.strip() + "\n",
                       year=2026, record={"effective_from": "2026-01-01", "effective_to": None, "status": "current"})


KB_SERIES = [
    _kb("kb_high_standard_overview", "High and Standard Options at a Glance", """
Both options are under contract 71-006 (FEHB) and 71-021 (PSHB). In-network
calendar year deductible: $500 Self, $1,000 Self Plus One or Self and
Family. High Option coinsurance 20%, Standard 25%. Specialist copay High
$45, Standard $50. Both offer mail order pharmacy and out-of-network
benefits.
"""),
    _kb("kb_hdhp_overview", "HDHP at a Glance", """
Contract 71-014 (FEHB) and 71-026 (PSHB). In-network deductible $1,800 Self
Only / $3,600 family tiers; the deductible applies to prescriptions except
preventive drugs. Coinsurance 5% in-network after the deductible. The Plan
funds an HSA (or HRA for ineligible members) monthly; see the HSA Basics
article.
"""),
    _kb("kb_elevate_overview", "Elevate and Elevate Plus at a Glance", """
Contract 71-018 (FEHB). Elevate Plus: $200 / $400 deductible, in-network
only except emergencies, mail order available. Elevate: $750 / $1,500
deductible, out-of-network at 50% of Plan allowance, no mail order. The
PSHB Elevate contract (71-022) was offered in 2025 only.
"""),
    _kb("kb_telehealth", "Telehealth", """
Telehealth with an in-network provider costs the same as an office visit of
the same specialty. Visits through the Plan's designated telehealth vendor
have no cost share under every option. Audio-only visits are covered
(Bulletin 2025-004).
"""),
    _kb("kb_urgent_care", "Urgent Care Visits", """
Urgent care visits use the urgent care copay in the member's brochure for
the plan year, not the emergency room cost share, even at hospital-owned
centers (Bulletin 2026-006). Telehealth urgent care through the designated
vendor is no cost share.
"""),
    _kb("kb_chiropractic_acupuncture", "Chiropractic and Acupuncture", """
Visits count against the annual limit in the member's brochure for the
plan year of service; the limit resets January 1. Evaluation billed the
same day as manipulation is bundled. PSHB members: the limit is per PSHB
plan brochure, and the three PSHB plans do not share one limit.
"""),
    _kb("kb_glp1_coverage", "GLP-1 Medications", """
GLP-1 drugs (Ozempic, Wegovy, Mounjaro, Zepbound) are on Formulary 2026 at
Tier 2 with prior authorization. Weight-management prescriptions are
reviewed under medical policy CP-0038; diabetes prescriptions need only
step therapy through metformin for new starts (Bulletin 2026-009).
Quantity limit: 30 days retail, 90 days mail order.
"""),
    _kb("kb_prior_auth_list", "Services Needing Prior Approval", """
Inpatient admissions (non-emergency), advanced imaging (MRI, CT, PET),
bariatric surgery, spinal fusion, home infusion, power mobility devices,
CPAP devices after the trial period, and weight-management GLP-1s. The
prescriber or facility submits the request; members cannot submit on a
provider's behalf. Decisions: 15 days non-urgent, 72 hours urgent.
"""),
    _kb("kb_member_appeals", "How a Member Appeals a Denial", """
A member has 180 days from the denial to appeal in writing. First-level
review takes up to 30 days (72 hours urgent pre-service). If the Plan
upholds the denial, the member may ask OPM to review within 90 days of the
Plan's decision. Clinical denials cite the medical policy version in effect
on the date of service.
"""),
    _kb("kb_medicare_partb_reimbursement", "Medicare Part B Premium Reimbursement", """
Annuitants with Medicare Part A and B under High Option, and from 2026
under Standard Option, may be reimbursed for the base Part B premium
through the account administrator. IRMAA surcharges are not reimbursed;
refer IRMAA questions to the Social Security Administration (Bulletin
2026-008).
"""),
    _kb("kb_pshb_medicare_requirement", "PSHB Medicare Part B Requirement", """
Postal annuitants who retired on or after 2025-01-01 and their Medicare
eligible family members must enroll in Medicare Part B to keep PSHB
coverage, with exceptions for those who retired before 2025 or live
abroad. Direct enrollment questions to the PSHB Navigator; the Plan does
not process Part B enrollment.
"""),
    _kb("kb_id_cards", "ID Cards", """
Replacement ID cards are mailed within 7 to 10 business days of the
request; a digital card is available immediately in the member portal.
One card is issued per enrollee; family members share the enrollee's
member ID.
"""),
    _kb("kb_out_of_country_care", "Care Outside the United States", """
Emergency and urgent care abroad is covered at the in-network cost share;
the member pays and submits an itemized bill with a translation and
currency conversion. Routine care abroad is covered at the out-of-network
share under High, Standard, and HDHP; Elevate Plus covers emergencies only.
"""),
    _kb("kb_dme", "Durable Medical Equipment", """
Rental converts to purchase after 13 months for rentals starting on or
after 2026-03-01 (10 months before); oxygen rents for 36 months (Bulletin
2026-007). Repairs are covered after the manufacturer warranty. Power
mobility devices need prior approval (policy CP-0006); hospital beds follow
CP-0026.
"""),
    _kb("kb_mental_health", "Mental Health and Substance Use", """
Outpatient mental health and substance use visits cost the same as a
primary care visit under every option; no referral is needed. Inpatient
admissions require notification within 2 business days. Transcranial
magnetic stimulation follows medical policy CP-0016.
"""),
    _kb("kb_timely_filing_members", "Claim Filing Deadline for Members", """
A claim must reach the Plan within 365 days of the date of service. When a
member pays a provider and files the claim, the same limit applies. Claims
where another plan paid first are accepted within 90 days of that plan's
explanation of benefits if later (Bulletin 2026-005).
"""),
]


_FORMULARY_ROWS_2025 = """
| Drug | Class | Tier | Prior Auth | Step Therapy | Quantity Limit |
|---|---|---|---|---|---|
| atorvastatin | statin | 1 | No | No | — |
| rosuvastatin | statin | 1 | No | No | — |
| simvastatin | statin | 1 | No | No | — |
| lisinopril | ACE inhibitor | 1 | No | No | — |
| losartan | ARB | 1 | No | No | — |
| amlodipine | calcium channel blocker | 1 | No | No | — |
| metoprolol succinate | beta blocker | 1 | No | No | — |
| hydrochlorothiazide | thiazide diuretic | 1 | No | No | — |
| metformin | biguanide | 1 | No | No | — |
| glipizide | sulfonylurea | 1 | No | No | — |
| insulin glargine (Basaglar) | long-acting insulin | 1 | No | No | — |
| insulin lispro (Admelog) | rapid-acting insulin | 1 | No | No | — |
| levothyroxine | thyroid | 1 | No | No | — |
| omeprazole | PPI | 1 | No | No | — |
| pantoprazole | PPI | 1 | No | No | — |
| sertraline | SSRI | 1 | No | No | — |
| escitalopram | SSRI | 1 | No | No | — |
| bupropion XL | antidepressant | 1 | No | No | — |
| trazodone | antidepressant | 1 | No | No | — |
| montelukast | leukotriene modifier | 1 | No | Yes | — |
| albuterol HFA | short-acting beta agonist | 1 | No | No | 2 inhalers / 30 days |
| budesonide/formoterol (generic) | ICS/LABA | 1 | No | No | — |
| fluticasone nasal | intranasal steroid | 1 | No | No | — |
| cetirizine | antihistamine | 1 | No | No | — |
| gabapentin | anticonvulsant | 1 | No | No | — |
| meloxicam | NSAID | 1 | No | No | — |
| cyclobenzaprine | muscle relaxant | 1 | No | No | — |
| prednisone | corticosteroid | 1 | No | No | — |
| amoxicillin | antibiotic | 1 | No | No | — |
| azithromycin | antibiotic | 1 | No | No | — |
| doxycycline | antibiotic | 1 | No | No | — |
| tamsulosin | alpha blocker | 1 | No | No | — |
| finasteride | 5-alpha reductase inhibitor | 1 | No | No | — |
| alendronate | bisphosphonate | 1 | No | No | — |
| warfarin | anticoagulant | 1 | No | No | — |
| clopidogrel | antiplatelet | 1 | No | No | — |
| Jardiance (empagliflozin) | SGLT2 inhibitor | 2 | Yes | No | — |
| Farxiga (dapagliflozin) | SGLT2 inhibitor | 2 | Yes | No | — |
| Januvia (sitagliptin) | DPP-4 inhibitor | 2 | No | Yes | — |
| Eliquis (apixaban) | anticoagulant | 2 | No | No | — |
| Breo Ellipta | ICS/LABA | 2 | No | No | 1 inhaler / 30 days |
| Symbicort | ICS/LABA | 2 | No | No | 1 inhaler / 30 days |
| Trelegy Ellipta | ICS/LAMA/LABA | 2 | No | Yes | 1 inhaler / 30 days |
| Ozempic (semaglutide) | GLP-1 receptor agonist | 2 | Yes | Yes | 30 days retail / 90 mail |
| Mounjaro (tirzepatide) | GIP/GLP-1 receptor agonist | 2 | Yes | Yes | 30 days retail / 90 mail |
| Lantus (insulin glargine) | long-acting insulin | 2 | No | No | — |
| Humira (adalimumab) | TNF inhibitor | 2 | Yes | Yes | specialty |
| Enbrel (etanercept) | TNF inhibitor | 2 | Yes | Yes | specialty |
| Entresto | ARNI | 2 | No | No | — |
| Xarelto (rivaroxaban) | anticoagulant | 3 | No | No | — |
| Dexilant | PPI | 3 | Yes | No | — |
| Wegovy (semaglutide) | GLP-1 receptor agonist (weight) | 3 | Yes | No | 30 days retail / 90 mail |
| Zepbound (tirzepatide) | GIP/GLP-1 receptor agonist (weight) | 3 | Yes | No | 30 days retail / 90 mail |
| Nurtec ODT | CGRP antagonist | 3 | Yes | Yes | 8 tablets / 30 days |
| Aimovig | CGRP antibody | 3 | Yes | Yes | specialty |
| Dupixent | IL-4/IL-13 antibody | 3 | Yes | Yes | specialty |
| Stelara | IL-12/23 antibody | 3 | Yes | Yes | specialty |
| Skyrizi | IL-23 antibody | 3 | Yes | Yes | specialty |
| Botox (onabotulinumtoxinA) | neurotoxin | Specialty | Yes | No | per policy CP-0036 |
| Repatha | PCSK9 inhibitor | Specialty | Yes | Yes | specialty |
"""

_FORMULARY_ROWS_2026 = (_FORMULARY_ROWS_2025
    .replace("| Symbicort | ICS/LABA | 2 | No | No | 1 inhaler / 30 days |",
             "| Symbicort | ICS/LABA | 3 | No | No | 1 inhaler / 30 days |")
    .replace("| Humira (adalimumab) | TNF inhibitor | 2 | Yes | Yes | specialty |",
             "| adalimumab-adaz (Humira biosimilar) | TNF inhibitor | 2 | Yes | Yes | specialty |\n"
             "| Humira (adalimumab) | TNF inhibitor | 3 | Yes | Yes | specialty |")
    .replace("| Wegovy (semaglutide) | GLP-1 receptor agonist (weight) | 3 | Yes | No | 30 days retail / 90 mail |",
             "| Wegovy (semaglutide) | GLP-1 receptor agonist (weight) | 2 | Yes | No | 30 days retail / 90 mail |")
    .replace("| Zepbound (tirzepatide) | GIP/GLP-1 receptor agonist (weight) | 3 | Yes | No | 30 days retail / 90 mail |",
             "| Zepbound (tirzepatide) | GIP/GLP-1 receptor agonist (weight) | 2 | Yes | No | 30 days retail / 90 mail |")
    .replace("| Januvia (sitagliptin) | DPP-4 inhibitor | 2 | No | Yes | — |",
             "| Januvia (sitagliptin) | DPP-4 inhibitor | 2 | No | Yes | — |\n"
             "| Rybelsus (oral semaglutide) | GLP-1 receptor agonist | 2 | Yes | Yes | — |"))


def _formulary(year: int, rows: str, notes: str) -> InternalDoc:
    body = (f"# Formulary {year} — Plan Year Reference\n\n"
            f"**Effective:** {year}-01-01 to {year}-12-31 · applies to every GEHA option; cost share by tier is in each option's brochure.\n\n"
            "Tier definitions: Tier 1 generic · Tier 2 preferred brand · Tier 3 non-preferred brand · Specialty per the specialty pharmacy program.\n"
            "Prior Auth: the prescriber submits a request before the first fill. Step Therapy: a listed alternative must be tried first for new starts.\n"
            + rows + "\n" + notes.strip() + "\n\n"
            "Drugs not listed are handled by the pharmacy benefit manager; direct members to the online formulary lookup for anything absent from this reference.\n")
    return InternalDoc(f"formulary/formulary_{year}.md", "employee", "formulary", f"Formulary {year}", body,
                       year=year, record={"formulary_year": year, "effective_from": f"{year}-01-01",
                                          "effective_to": f"{year + 1}-01-01" if year < 2026 else None,
                                          "status": "current" if year == 2026 else "superseded"})


FORMULARY_SERIES = [
    _formulary(2025, _FORMULARY_ROWS_2025, """
Notes for 2025: GLP-1 receptor agonists prescribed for weight management
(Wegovy, Zepbound) are Tier 3 with prior authorization under medical policy
CP-0038; the same molecules prescribed for type 2 diabetes (Ozempic,
Mounjaro) are Tier 2 with step therapy through metformin.
"""),
    _formulary(2026, _FORMULARY_ROWS_2026, """
Notes for 2026: Wegovy and Zepbound move to Tier 2 (prior authorization
under CP-0038 version 3 continues). Rybelsus added at Tier 2. Humira
biosimilar adalimumab-adaz is Tier 2; reference Humira moves to Tier 3.
Symbicort moves to Tier 3 with generic budesonide/formoterol at Tier 1.
"""),
]


def _updates(year: int, quarter: int, issued: str, body: str, effective_to: str | None) -> InternalDoc:
    return InternalDoc(f"formulary/formulary_updates_{year}_q{quarter}.md", "employee", "formulary",
                       f"Formulary Updates {year} Q{quarter}",
                       f"# Formulary Updates — {year} Q{quarter}\n\n**Issued:** {issued} · pharmacy services\n\n" + body.strip() + "\n",
                       year=year, record={"formulary_year": year, "quarter": quarter, "effective_from": issued,
                                          "effective_to": effective_to,
                                          "status": "superseded" if effective_to else "current"})


FORMULARY_UPDATES = [
    _updates(2025, 1, "2025-01-06", """
- Farxiga added at Tier 2 with prior authorization.
- Trelegy Ellipta added at Tier 2 with step therapy through an ICS/LABA.
- montelukast added to the step-therapy protocol for new starts.
""", "2025-04-07"),
    _updates(2025, 2, "2025-04-07", """
- Aimovig quantity limit set to one autoinjector per 28 days.
- Repatha step therapy: a high-intensity statin plus ezetimibe for 90 days.
""", "2025-07-07"),
    _updates(2025, 3, "2025-07-07", """
- Nurtec ODT added at Tier 3 with prior authorization and a limit of 8
  tablets per 30 days.
- insulin glargine (Basaglar) confirmed Tier 1 under the HDHP preventive
  drug benefit; the deductible does not apply.
""", "2025-10-06"),
    _updates(2025, 4, "2025-10-06", """
- Symbicort moves Tier 2 to Tier 3 effective 2025-11-01; generic
  budesonide/formoterol is available at Tier 1.
- Wegovy and Zepbound will move to Tier 2 on 2026-01-01 (see Formulary 2026).
""", "2026-01-05"),
    _updates(2026, 1, "2026-01-05", """
- Humira biosimilar adalimumab-adaz added at Tier 2 with prior authorization;
  reference Humira moves to Tier 3.
- Rybelsus added at Tier 2 with prior authorization and step therapy.
- Wegovy and Zepbound now Tier 2 (prior authorization under CP-0038 v3).
""", "2026-04-06"),
    _updates(2026, 2, "2026-04-06", """
- Skyrizi quantity limit aligned with the labeled maintenance interval.
- Dexilant prior authorization now requires a 60-day trial of a Tier 1 PPI.
""", "2026-07-06"),
    _updates(2026, 3, "2026-07-06", """
- Entresto confirmed Tier 2 without prior authorization.
- Step therapy for Trelegy Ellipta now satisfied by any Tier 1 or Tier 2
  ICS/LABA for 60 days.
""", None),
]

SERIES_DOCS = BULLETIN_SERIES + SOP_ARCHIVE + SOP_SERIES + KB_SERIES + FORMULARY_SERIES + FORMULARY_UPDATES
