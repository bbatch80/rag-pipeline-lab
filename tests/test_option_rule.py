"""The option rule: a chunk's section heading names which of a brochure's
options it belongs to (derived metadata, never hand-tagged); a question that
names an option keeps chunks under that option's headings or under none, and
drops the other option's — the Elevate Plus mail-order page is the distractor
for an Elevate question. Offline."""

from raglab.metadata import section_options
from raglab.retrieval import _filters
from raglab.router import Reading, Route, asked_options, enforce, route

E = ("Elevate", "Elevate Plus")
HS = ("High", "Standard")


def test_headings_name_the_option_they_belong_to():
    assert section_options("Standard Option", HS) == ["Standard"]
    assert section_options("Changes to both High and Standard Option", HS) == ["High", "Standard"]
    assert section_options("Section 4. Your Costs for Covered Services", HS) == []
    assert section_options("Summary of Benefits for the Elevate Plus Option of the Government Employees Health Association, Inc. 2026", E) == ["Elevate Plus"]
    assert section_options("Summary of Benefits for the Elevate Option of the Government Employees Health Association, Inc. 2026", E) == ["Elevate"]
    assert section_options("Elevate Plus and Elevate Options", E) == ["Elevate", "Elevate Plus"]
    assert section_options("How to use CVS Caremark Mail Service Pharmacy for Elevate Plus", E) == ["Elevate Plus"]
    assert section_options("Elevate", E) == ["Elevate"]


def test_common_words_need_the_word_option_beside_them_and_single_option_brochures_are_never_tagged():
    assert section_options("Standard of care", HS) == []
    assert section_options("High Deductible Health Plan", ("HDHP",)) == []


def test_the_question_names_its_option():
    assert route("Does the FEHB Elevate have a mail-order drug benefit?").options == ("Elevate",)
    assert route("What is the Elevate Plus deductible?").options == ("Elevate Plus",)
    assert route("What would my deductible be on the standard plan?").options == ("Standard",)
    assert route("Compare High and Standard Option copays").options == ("Standard", "High")
    assert route("What are the costs for physical therapy?").options == ()
    assert asked_options("Elevate vs Elevate Plus: which has mail order?", ("elevate", "elevate_plus")) == ("Elevate", "Elevate Plus")
    assert enforce(Reading(program="FEHB", options=("elevate_plus",), years=(2026,), origin="model")).options == ("Elevate Plus",)


def test_filter_keeps_the_asked_option_and_untagged_chunks_only():
    where, params = _filters(Route(scope="in_scope", years=(2026,), options=("Elevate",)))
    assert "section_options" in where and ["Elevate"] in params
    where, params = _filters(Route(scope="in_scope", years=(2026,)))
    assert "section_options" not in where
