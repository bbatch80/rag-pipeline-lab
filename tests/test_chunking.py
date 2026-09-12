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


def _table(rows, page=1):
    lines = ["| Benefit | High Option | Standard Option |", "| --- | --- | --- |"] + [f"| {r} | ${i} | ${i+1} |" for i, r in enumerate(rows)]
    return Element(text="\n".join(lines), category="table", page=page)


def test_a_table_is_its_own_chunk_and_never_merges_into_prose():
    """Table-aware chunking (2026-09-12): the row 'Specialist | $45 | $50'
    stays with its column headers, apart from the prose around it."""
    els = [Element(text="Section 5", category="title", page=1),
           Element(text="Some short prose.", category="text", page=1),
           _table(["Specialist", "Primary care"]),
           Element(text="More short prose.", category="text", page=1)]
    chunks = chunk_elements(els, merge_under=250)
    tables = [c for c in chunks if "table" in c.categories]
    assert len(tables) == 1 and tables[0].text.startswith("| Benefit | High Option")
    assert "Specialist" in tables[0].text and "prose" not in tables[0].text
    assert all("table" not in c.categories or c is tables[0] for c in chunks)


def test_an_oversized_table_splits_on_rows_and_repeats_its_headers():
    rows = [f"Benefit row number {i} with a long description to fill space" for i in range(60)]
    chunks = chunk_elements([_table(rows)], hard_max=800)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("| Benefit | High Option | Standard Option |\n| --- |"), "every piece carries the header rows"
        assert all(line.startswith("|") and line.endswith("|") for line in c.text.split("\n")), "no row is cut mid-way"
    joined = "\n".join(c.text for c in chunks)
    assert all(f"Benefit row number {i} " in joined for i in range(60)), "no row lost"
