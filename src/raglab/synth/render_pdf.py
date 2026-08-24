"""Render the PDF-bound clinical notes (fpdf2, Helvetica — content is
ASCII-normalized at injection time, so no glyph surprises)."""

from pathlib import Path

from raglab import config
from raglab.synth.notes import PDF_SRC_DIR

NOTES_PDF_DIR = config.REPO_ROOT / "data" / "internal" / "notes_pdf"


def render_all(
    src_dir: Path = PDF_SRC_DIR, out_dir: Path = NOTES_PDF_DIR
) -> int:
    from datetime import datetime, timezone

    from fpdf import FPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob("*.txt")):
        pdf = FPDF()
        # Pinned creation date: renders must be byte-deterministic or every
        # synth rerun re-triggers ingest/embed for all PDFs (hash contract).
        pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        for line in src.read_text().splitlines():
            if not line.strip():
                pdf.ln(4)
                continue
            style = "B" if line.isupper() or line.startswith(("CLINIC", "DISCHARGE")) else ""
            pdf.set_font("Helvetica", style, 10)
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(out_dir / (src.stem + ".pdf")))
        count += 1
    return count
