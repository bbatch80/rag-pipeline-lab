"""Chunker contract tests: bounds, merges, heading behavior."""

from raglab.chunking import chunk_elements
from raglab.parsing.base import Element


def title(text):
    return Element(text=text, category="title", page=1)


def text(body):
    return Element(text=body, category="text", page=1)


def test_no_chunk_exceeds_hard_max():
    elements = [title("Section A")] + [text("word " * 300) for _ in range(5)]
    chunks = chunk_elements(elements, hard_max=1000, soft_max=800, merge_under=100)
    assert all(len(c.text) <= 1000 for c in chunks)


def test_oversized_element_is_split_on_whitespace():
    elements = [text("word " * 1000)]
    chunks = chunk_elements(elements, hard_max=1000, soft_max=800, merge_under=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    assert not any(c.text.startswith("ord") for c in chunks), "split mid-word"


def test_small_chunks_merge_into_neighbor():
    elements = [
        title("Section A"),
        text("a" * 500),
        title("Section B"),
        text("tiny"),
    ]
    chunks = chunk_elements(elements, hard_max=2000, soft_max=1500, merge_under=250)
    assert len(chunks) == 1, "trailing fragment should fold into previous chunk"


def test_heading_closes_only_accumulated_chunks():
    # Three consecutive titles with no body: must NOT produce three chunks.
    elements = [title("A"), title("B"), title("C"), text("body " * 100)]
    chunks = chunk_elements(elements, hard_max=2000, soft_max=1500, merge_under=250)
    assert len(chunks) == 1


def test_section_label_comes_from_heading():
    elements = [
        title("Deductibles"),
        text("x" * 400),
        title("Copayments"),
        text("y" * 400),
    ]
    chunks = chunk_elements(elements, hard_max=2000, soft_max=1500, merge_under=250)
    assert [c.section for c in chunks] == ["Deductibles", "Copayments"]


def test_small_leading_chunk_merges_forward():
    elements = [text("cover page"), title("Section A"), text("z" * 600)]
    chunks = chunk_elements(elements, hard_max=2000, soft_max=1500, merge_under=250)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("cover page")
