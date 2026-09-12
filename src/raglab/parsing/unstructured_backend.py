"""Unstructured backend: `fast` (text layer only; the production default
since the Phase 0 bake-off) or `hi_res` (layout detection + table structure;
needs the Tesseract binary and two local model weights). A hi_res Table
element carries its cells as HTML; it is rendered to a markdown table so a
row keeps its label and column headers in one piece of text."""

import os
from html.parser import HTMLParser
from pathlib import Path

# No telemetry from a data pipeline.
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("SCARF_NO_ANALYTICS", "true")

from raglab.parsing.base import Element

# Repeated page furniture pollutes embeddings; drop before chunking.
_DROP = {"Header", "Footer", "PageBreak"}

_CATEGORY_MAP = {
    "Title": "title",
    "Table": "table",
    "ListItem": "list_item",
}


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows, self._row, self._cell, self._in = [], None, None, False
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self._row = []
        elif tag in ("td", "th"): self._cell, self._in = [], True
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split())); self._cell, self._in = None, False
        elif tag == "tr" and self._row is not None:
            if any(self._row): self.rows.append(self._row)
            self._row = None
    def handle_data(self, data):
        if self._in: self._cell.append(data)


def table_html_to_markdown(html: str) -> str:
    """A markdown table from an HTML table: header row, separator, rows.
    Cells never lose their row; a chunker that keeps this text whole keeps
    the table whole."""
    parser = _Rows(); parser.feed(html or "")
    rows = [r for r in parser.rows if r]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(c.replace("|", "/") for c in rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(c.replace("|", "/") for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


class UnstructuredBackend:
    def __init__(self, strategy: str = "fast"):
        self.strategy = strategy
        self.name = "unstructured-fast" if strategy == "fast" else f"unstructured-{strategy.replace('_', '')}"

    def parse(self, path: Path) -> list[Element]:
        from unstructured.partition.pdf import partition_pdf

        kwargs = {"strategy": self.strategy, "languages": ["eng"]}
        if self.strategy == "hi_res":
            kwargs["infer_table_structure"] = True
        raw = partition_pdf(filename=str(path), **kwargs)
        elements = []
        for el in raw:
            if el.category in _DROP:
                continue
            text = (el.text or "").strip()
            if el.category == "Table" and getattr(el.metadata, "text_as_html", None):
                text = table_html_to_markdown(el.metadata.text_as_html) or text
            if not text:
                continue
            elements.append(
                Element(
                    text=text,
                    category=_CATEGORY_MAP.get(el.category, "text"),
                    page=getattr(el.metadata, "page_number", None),
                )
            )
        return elements
