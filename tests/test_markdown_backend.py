"""Markdown/CSV backend normalization contract."""

from raglab.chunking import chunk_elements
from raglab.parsing.base import CATEGORIES
from raglab.parsing.markdown_backend import CsvBackend, MarkdownBackend


def test_markdown_backend_normalizes(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# SOP: Example\n\n"
        "Intro paragraph explaining the procedure\n"
        "continues on a second line.\n\n"
        "- first step\n- second step\n\n"
        "| Drug | Tier |\n|---|---|\n| atorvastatin | 1 |\n\n"
        "## Subsection\n\nClosing text.\n"
    )
    elements = MarkdownBackend().parse(doc)
    categories = [e.category for e in elements]
    assert all(c in CATEGORIES for c in categories)
    assert categories.count("title") == 2
    assert categories.count("list_item") == 2
    assert categories.count("table") == 1
    joined = next(e.text for e in elements if e.category == "text")
    assert "continues on a second line" in joined, "paragraph lines must join"
    assert chunk_elements(elements), "chunker must accept the stream"


def test_csv_backend_rows_are_self_contained(tmp_path):
    doc = tmp_path / "rates.csv"
    doc.write_text("plan,code,premium\n71-006,311,195.29\n71-006,313,432.95\n")
    elements = CsvBackend().parse(doc)
    assert elements[0].category == "title"
    rows = [e for e in elements if e.category == "table"]
    assert len(rows) == 2
    assert "plan: 71-006" in rows[0].text and "premium: 195.29" in rows[0].text, (
        "each row must carry header context"
    )
