"""The golden file is well-formed: every item carries both axes, names a
group the taxonomy knows, declares its identity where it needs one, and
points its evidence at folders the corpus has. Structural only — the
corpus itself is checked by the eval run."""

import re

from raglab import ablation, checks, taxonomy
from raglab.mcp_server import IDENTITIES

ITEMS = ablation.load_golden()
INTERNAL_FOLDERS = ("appeals/", "appeals_pdf/", "bulletins/", "calls/", "carrier_letters/", "formulary/", "kb/",
                    "notes/", "notes_pdf/", "policies/", "rates/", "sops/")


def test_every_item_has_the_required_fields_and_a_known_group():
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    for item in ITEMS:
        assert {"id", "category", "work_category", "question", "answer"} <= set(item), item["id"]
        assert item["category"] in taxonomy.GROUPS, (item["id"], item["category"])
        assert int(item["work_category"]) in taxonomy.WORK_CATEGORIES, item["id"]
        assert re.fullmatch(rf"{item['category']}-\d+", item["id"]), f"{item['id']} is not <group>-<n>"


def test_identities_and_screens_are_known():
    for item in ITEMS:
        for key in ("persona", "persona_allow", "persona_deny", "persona_deny_2"):
            if item.get(key):
                assert item[key] in IDENTITIES, (item["id"], key, item[key])
        if item.get("module"):
            assert item["module"] in ("ask", "agent_assist", "appeals_workbench", "care_management", "analyst_view"), item["id"]


def test_evidence_points_at_real_folders_and_editions():
    for item in ITEMS:
        for key in ("sources", "exclude_sources"):
            for src in item.get(key) or []:
                if "internal" in src:
                    assert src["internal"].startswith(INTERNAL_FOLDERS), (item["id"], src["internal"])
                else:
                    assert re.fullmatch(r"\d\d-\d\d\d", src["plan_code"]) and 2021 <= int(src["year"]) <= 2026, (item["id"], src)
                    assert src.get("printed_pages"), (item["id"], "brochure source without pages")


def test_every_item_can_be_judged():
    """Each item declares at least one check, is guardrail-shaped, or lists
    sources for the hit@5 fallback — never nothing."""
    for item in ITEMS:
        judged = (checks.declared_checks(item) or item.get("sources") or item.get("persona_allow") or item.get("persona_deny")
                  or item.get("control_question") or item.get("expected_trigger"))
        assert judged, f"{item['id']} declares nothing the eval can check"


def test_expected_legs_normalize():
    for item in ITEMS:
        for leg in item.get("expected_legs") or []:
            norm = checks.normalize_leg(leg)
            assert norm["kind"] in ("member_query", "doc_probe"), (item["id"], leg)
