"""Authored internal-tier documents: SOPs and claims bulletins (employee),
formulary + CSR knowledge base (employee), structured rates CSV (public).

Content interprets the real brochures — figures match the 2026 corpus.
The formulary deliberately contains NO GLP-1 drugs (Ozempic stays a valid
abstention question); Jardiance is the answerable mirror twin.
formulary_core.md is golden-anchored: the churn simulator must never touch it.
"""

from dataclasses import dataclass
from pathlib import Path

from raglab import config

INTERNAL_DIR = config.REPO_ROOT / "data" / "internal"

# Documents the churn simulator must never mutate or delete — everything the
# golden set cites as a source (eval labels are assertions about the corpus).
GOLDEN_ANCHORED = {
    "formulary/formulary_core.md",
    "kb/kb_mail_order.md",
    "kb/kb_specialist_visits.md",
}


@dataclass(frozen=True)
class InternalDoc:
    relpath: str
    acl_tag: str
    doc_type: str
    title: str
    content: str


SOPS = [
    InternalDoc(
        "sops/sop_deductible_verification.md", "employee", "sop",
        "SOP: HDHP Deductible Verification",
        """# SOP: HDHP Deductible Verification

**Applies to:** Claims examiners, tier 1-2
**Effective:** 2026-01-01

## Procedure

1. Confirm member enrollment option. For HDHP (71-014) the 2026 in-network
   calendar year deductible is $1,800 Self Only / $3,600 Self Plus One or
   Self and Family. Out-of-network: $4,500 / $9,000.
2. Preventive care services bypass the deductible entirely when rendered
   in-network — do not apply deductible logic to claims coded preventive.
3. Deductible accumulators reset January 1. Claims spanning the year boundary
   apply to the year of the service date, not the processing date.
4. If the member changed enrollment options mid-year, credit covered expenses
   already applied under the prior option toward the new option's deductible.
5. Route disputes over accumulator balances to a senior examiner — do not
   adjust accumulators manually.
""",
    ),
    InternalDoc(
        "sops/sop_prior_authorization.md", "employee", "sop",
        "SOP: Prior Authorization Intake",
        """# SOP: Prior Authorization Intake

**Applies to:** Utilization review intake
**Effective:** 2026-01-01

## Services requiring prior Plan approval

Inpatient facility admissions, certain imaging (CT, MRI, PET in specified
contexts), and services listed in Section 3 of the applicable brochure.

## Procedure

1. Verify the requesting provider is identified and the member ID is valid.
2. Non-urgent care requests: decision within 15 calendar days.
3. Urgent care claims: decision within 72 hours of receipt.
4. Emergency inpatient admissions do not require pre-approval; notification
   is required within 2 business days following the admission.
5. Missing clinical documentation: pend the request and notify the provider;
   do not deny solely for missing documentation on first pass.
""",
    ),
    InternalDoc(
        "sops/sop_claims_escalation.md", "employee", "sop",
        "SOP: Claims Escalation Thresholds",
        """# SOP: Claims Escalation Thresholds

**Applies to:** Claims examiners
**Effective:** 2026-01-01

## Escalation rules

- Claims with billed amount over $10,000: senior examiner review required
  before adjudication.
- Claims involving out-of-network emergency services: verify No Surprises Act
  balance-billing protections before applying member cost share.
- Potential coordination-of-benefits (other coverage indicated): route to the
  COB unit; do not adjudicate as primary without verification.
- Three or more corrected claims from the same provider in 30 days: flag to
  provider relations.
- Suspected fraud indicators: route directly to SIU. Do not contact the
  provider or member.
""",
    ),
    InternalDoc(
        "sops/sop_cob_medicare.md", "employee", "sop",
        "SOP: Coordination of Benefits with Medicare",
        """# SOP: Coordination of Benefits with Medicare

**Applies to:** COB unit
**Effective:** 2026-01-01

## Order of benefits

1. For active federal employees age 65+, the Plan pays primary and Medicare
   pays secondary.
2. For annuitants enrolled in Medicare Part A and B, Medicare pays primary
   and the Plan pays secondary.
3. When the Plan is secondary to Medicare and the provider accepts Medicare
   assignment, member cost share for covered services is generally reduced —
   apply the brochure Section 9 coordination rules.
4. Verify Medicare enrollment status via the eligibility file before
   adjudicating any claim flagged age 65 or over.
""",
    ),
]

BULLETINS = [
    InternalDoc(
        "bulletins/bulletin_2026_001.md", "employee", "bulletin",
        "Claims Bulletin 2026-001: Benefit Changes",
        """# Claims Bulletin 2026-001 — 2026 Benefit Changes (71-006)

**Issued:** 2026-01-05 · **Audience:** all examiners

Key adjudication changes for the GEHA Benefit Plan effective 2026-01-01:

- In-network deductible increased to $500 Self / $1,000 Self Plus One or
  Self and Family (was $350 / $700). Verify accumulator configuration.
- High Option in-network coinsurance is now 20% (was 10%). Standard Option
  is now 25% (was 15%).
- High Option inpatient copay structure changed: $200 per day up to 5 days
  (was $100 per admission). Per-day logic applies to admissions on or after
  2026-01-01, based on admission date.
- Specialist office visit copays: High $45, Standard $50.
""",
    ),
    InternalDoc(
        "bulletins/bulletin_2026_002.md", "employee", "bulletin",
        "Claims Bulletin 2026-002: Emergency Room Cost Share",
        """# Claims Bulletin 2026-002 — ER Cost Share Handling

**Issued:** 2026-01-12 · **Audience:** all examiners

- High Option ER cost share is 30% of Plan allowance after deductible
  (was 15%). Standard Option is 35% (was 20%).
- The accidental-injury cost-share waiver window narrowed to 48 hours from
  the accident (was 72 hours). Apply the waiver only when services were
  rendered within 48 hours — use the accident date/time on the claim.
- ER claims for accidental injury outside the window adjudicate at the
  standard ER cost share; do not deny.
""",
    ),
    InternalDoc(
        "bulletins/bulletin_2026_003.md", "employee", "bulletin",
        "Claims Bulletin 2026-003: PSHB Claims Routing",
        """# Claims Bulletin 2026-003 — PSHB Program Claims Routing

**Issued:** 2026-01-20 · **Audience:** intake, all examiners

- Postal Service Health Benefits (PSHB) members are enrolled under separate
  contracts: 71-021 (High/Standard), 71-026 (HDHP). Verify program before
  applying benefit rules — FEHB and PSHB accumulators are separate.
- The PSHB Indemnity option (71-022) existed for 2025 only; claims with 2025
  service dates adjudicate under the 2025 PSHB brochure. There is no 2026
  71-022 plan.
- Misrouted FEHB/PSHB claims: reroute via the enrollment reconciliation
  queue, do not deny for wrong program.
""",
    ),
]

FORMULARY = [
    InternalDoc(
        "formulary/formulary_core.md", "employee", "formulary",
        "Formulary Reference: Core Tiers",
        """# Formulary Reference — Core Tiers (2026)

**GOLDEN-ANCHORED — churn-exempt.** CSR reference for tier inquiries.

Tier definitions: Tier 1 generic · Tier 2 preferred brand · Tier 3
non-preferred brand · Specialty per specialty pharmacy program.

| Drug | Class | Tier | Prior Auth |
|---|---|---|---|
| atorvastatin | statin | 1 | No |
| lisinopril | ACE inhibitor | 1 | No |
| metformin | biguanide | 1 | No |
| Jardiance (empagliflozin) | SGLT2 inhibitor | 2 | Yes |
| Eliquis (apixaban) | anticoagulant | 2 | No |
| Breo Ellipta | ICS/LABA | 2 | No |
| Xarelto (rivaroxaban) | anticoagulant | 3 | No |
| Dexilant | PPI | 3 | Yes |

Drugs not listed here or in formulary updates are handled by the PBM;
direct members to the CVS Caremark formulary lookup for anything absent
from this reference.
""",
    ),
    InternalDoc(
        "formulary/formulary_updates.md", "employee", "formulary",
        "Formulary Updates Log",
        """# Formulary Updates Log (2026)

Rolling updates issued by pharmacy services. Most recent first.

- 2026-01: Humira biosimilars (adalimumab-adaz) added at Tier 2 with PA;
  reference-product Humira moves to Tier 3.
- 2026-01: insulin glargine (Basaglar) confirmed Tier 1 under the preventive
  drug benefit for HDHP members — deductible does not apply.
- 2025-11: Symbicort moved Tier 2 -> Tier 3; generic budesonide/formoterol
  available Tier 1.
- 2025-10: montelukast added to step-therapy protocol for new starts.
""",
    ),
]

KB_ARTICLES = [
    InternalDoc(
        "kb/kb_deductible_faq.md", "employee", "kb",
        "CSR KB: Deductible FAQ",
        """# CSR KB: Deductible FAQ (2026)

**Q: What is the deductible this year?** High/Standard: $500 Self, $1,000
family tiers, in-network. HDHP: $1,800 / $3,600 in-network. Elevate Plus:
$200 / $400. Elevate: $750 / $1,500.

**Q: Do copays count toward the deductible?** No. Copayments and coinsurance
do not count toward any deductible.

**Q: Member changed options mid-year?** Expenses already applied to the old
option's deductible carry to the new option's deductible.
""",
    ),
    InternalDoc(
        "kb/kb_specialist_visits.md", "employee", "kb",
        "CSR KB: Specialist Visit Copays",
        """# CSR KB: Specialist Visit Copays (2026)

In-network specialist office visit cost share by option: High $45 ·
Standard $50 · HDHP 5% after deductible · Elevate $30 · Elevate Plus $50.

No referral is required to see a specialist under any GEHA option.
""",
    ),
    InternalDoc(
        "kb/kb_mail_order.md", "employee", "kb",
        "CSR KB: Mail Order Pharmacy",
        """# CSR KB: Mail Order Pharmacy (2026)

Mail order (90-day supply) is available under High, Standard, HDHP, and
Elevate Plus. **The Elevate option does not offer mail order** — retail
network only; advise Elevate members accordingly.

High Option mail order: generic $25; preferred brand 25% up to $500;
non-preferred 40% up to $800.
""",
    ),
    InternalDoc(
        "kb/kb_hdhp_hsa.md", "employee", "kb",
        "CSR KB: HDHP HSA Basics",
        """# CSR KB: HDHP HSA Basics (2026)

- 2026 plan pass-through: $83.33/month Self Only, $166.66/month Self Plus
  One or Self and Family.
- 2026 IRS contribution limits: $4,400 Self Only, $8,750 family tiers
  (pass-through counts toward the limit).
- Members not HSA-eligible receive an HRA instead: up to $1,000 / $2,000
  annually.
- HSA is portable; HRA is not.
""",
    ),
    InternalDoc(
        "kb/kb_elevate_network.md", "employee", "kb",
        "CSR KB: Elevate Options Network Rules",
        """# CSR KB: Elevate Options Network Rules (2026)

Elevate Plus has NO out-of-network benefits except emergency care.
Elevate covers out-of-network at 50% of Plan allowance plus balance billing
exposure. Always verify the provider is in-network before advising members
on either Elevate option.
""",
    ),
    InternalDoc(
        "kb/kb_preventive_care.md", "employee", "kb",
        "CSR KB: Preventive Care",
        """# CSR KB: Preventive Care (2026)

In-network preventive services are covered at no cost share under every
option, and are not subject to the deductible (including HDHP). Breast
screening is covered for ages 40-74 effective 2026 (previously 50-74).
""",
    ),
    InternalDoc(
        "kb/kb_claims_status.md", "employee", "kb",
        "CSR KB: Claims Status Scripts",
        """# CSR KB: Claims Status Scripts

Standard processing: most clean claims adjudicate within 15 business days.
If a claim is pended for documentation, the provider was notified — advise
members no action is needed unless contacted. Escalate claims older than
30 days to a supervisor queue.
""",
    ),
    InternalDoc(
        "kb/kb_pshb_transition.md", "employee", "kb",
        "CSR KB: PSHB Transition Questions",
        """# CSR KB: PSHB Transition Questions

Postal employees and annuitants moved to PSHB plans effective 2025.
GEHA PSHB options: High/Standard (71-021) and HDHP (71-026). The PSHB
Indemnity option (71-022) was offered in 2025 only. FEHB and PSHB are
separate programs with separate brochures and accumulators.
""",
    ),
]

RATES_CSV = InternalDoc(
    "rates/rates_2026.csv", "public", "rates",
    "2026 Rate Information (structured)",
    """plan_code,option,enrollment_type,enrollment_code,biweekly_govt,biweekly_your_share,monthly_govt,monthly_your_share
71-006,High,Self Only,311,324.76,195.29,703.65,423.13
71-006,High,Self Plus One,313,711.17,432.95,1540.87,938.06
71-006,High,Self and Family,312,778.03,525.18,1685.73,1137.89
71-006,Standard,Self Only,314,260.24,86.75,563.86,187.95
71-006,Standard,Self Plus One,316,559.55,186.51,1212.35,404.11
71-006,Standard,Self and Family,315,694.34,231.45,1504.41,501.47
""",
)

ALL_DOCS = SOPS + BULLETINS + FORMULARY + KB_ARTICLES + [RATES_CSV]


def write_all(base_dir: Path = INTERNAL_DIR) -> int:
    for doc in ALL_DOCS:
        path = base_dir / doc.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.content)
    return len(ALL_DOCS)
