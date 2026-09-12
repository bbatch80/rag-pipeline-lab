"""The table-row metric and the bake-off scratch path (Phase 3.5)."""
import pytest

from raglab import tablebakeoff
from raglab.parsing.unstructured_backend import table_html_to_markdown

pytestmark = pytest.mark.readonly


def test_row_intact_needs_value_row_and_column_in_one_chunk():
    cell = {"value": "$45", "row": "Specialist", "column": "High Option"}
    whole = ["| Benefit | High Option | Standard Option |\n| --- | --- | --- |\n| Specialist | $45 | $50 |"]
    shredded = ["High Option", "Standard Option", "Specialist", "$45", "$50"]
    assert tablebakeoff.row_intact(whole, cell)
    assert not tablebakeoff.row_intact(shredded, cell)
    assert not tablebakeoff.row_intact(["Specialist $45"], cell), "the column header is part of the row's meaning"
    prose = ["Under the High Option you see a Specialist for a copayment. The generic drug copay is $45 at retail."]
    assert not tablebakeoff.row_intact(prose, cell), "three strings in one paragraph are not a row"
    assert tablebakeoff.row_intact(["Benefit: Specialist | High Option $45 | Standard $50"], cell), "a plain-text row on one line counts"


def test_html_table_becomes_a_markdown_table_with_headers():
    md = table_html_to_markdown("<table><tr><th>Benefit</th><th>High</th></tr><tr><td>Specialist</td><td>$45</td></tr></table>")
    assert md.splitlines()[0] == "| Benefit | High |" and md.splitlines()[2] == "| Specialist | $45 |"
    assert table_html_to_markdown("") == ""


def test_labeled_cells_are_well_formed():
    cells = tablebakeoff.load_cells()
    assert cells and all({"plan_code", "year", "value", "row", "column"} <= set(c) for c in cells)


def test_scratch_report_scores_each_parser_separately(db):
    db.execute(tablebakeoff.EVAL_SCHEMA_PATH.read_text())
    db.execute("DELETE FROM chunks_bakeoff")
    src = "data/raw/2026/71-018.pdf"
    db.execute("INSERT INTO chunks_bakeoff (parser, source_path, chunk_index, section, pages, categories, content) VALUES "
               "('whole', %s, 0, 'Summary', '{127}', '{table}', %s), ('shred', %s, 0, 'Elevate', '{127}', '{text}', 'Specialist'), "
               "('shred', %s, 1, 'Elevate', '{127}', '{text}', '$30')",
               (src, "| Benefit | Elevate |\n| --- | --- |\n| Specialist | $30 |", src, src))
    r = tablebakeoff.report(db)
    assert r["whole"]["rows_intact"] == 1 and r["whole"]["table_chunks"] == 1
    assert r["shred"]["rows_intact"] == 0 and "T6" in r["shred"]["rows_missing"]
    assert r["whole"]["cells_scored"] == r["shred"]["cells_scored"] == 3, "only the parsed document's cells are scored"
