"""Backend normalization contract: every backend parses the same PDF into
valid normalized Elements that the chunker accepts."""

import pytest

pytestmark = pytest.mark.slow

from raglab.chunking import chunk_elements
from raglab.parsing.base import CATEGORIES


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    from fpdf import FPDF

    pdf = FPDF()
    for page in range(2):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Section Heading {page + 1}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        for i in range(10):
            pdf.multi_cell(
                0,
                6,
                "The calendar year deductible applies before the plan pays "
                "benefits. Out-of-pocket costs accumulate toward the annual "
                f"limit as described in paragraph {i + 1}. ",
                new_x="LMARGIN",
                new_y="NEXT",
            )
    path = tmp_path_factory.mktemp("pdfs") / "sample.pdf"
    pdf.output(str(path))
    return path


def _assert_normalized(elements):
    assert elements, "backend returned zero elements"
    assert all(e.text.strip() for e in elements)
    assert all(e.category in CATEGORIES for e in elements)
    assert any(e.page is not None for e in elements)
    chunks = chunk_elements(elements)
    assert chunks, "chunker produced nothing from parsed elements"
    assert all(c.section is not None for c in chunks)


def test_unstructured_backend_normalizes(sample_pdf):
    from raglab.parsing.unstructured_backend import UnstructuredBackend

    _assert_normalized(UnstructuredBackend().parse(sample_pdf))

