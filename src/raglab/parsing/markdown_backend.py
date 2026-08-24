"""Markdown/plain-text backend: the internal tier's fast path. Line-based,
no dependencies; normalizes to the same Element stream as the PDF backends."""

from pathlib import Path

from raglab.parsing.base import Element


class MarkdownBackend:
    name = "markdown"

    def parse(self, path: Path) -> list[Element]:
        elements: list[Element] = []
        paragraph: list[str] = []
        table: list[str] = []

        def flush_paragraph():
            if paragraph:
                elements.append(Element(text=" ".join(paragraph), category="text", page=1))
                paragraph.clear()

        def flush_table():
            if table:
                elements.append(Element(text="\n".join(table), category="table", page=1))
                table.clear()

        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                flush_paragraph()
                table.append(stripped)
                continue
            flush_table()
            if not stripped:
                flush_paragraph()
            elif stripped.startswith("#"):
                flush_paragraph()
                elements.append(
                    Element(text=stripped.lstrip("#").strip(), category="title", page=1)
                )
            elif stripped.startswith(("- ", "* ")) or stripped[:2].rstrip(".").isdigit():
                flush_paragraph()
                elements.append(
                    Element(text=stripped.lstrip("-* ").strip(), category="list_item", page=1)
                )
            else:
                paragraph.append(stripped)
        flush_paragraph()
        flush_table()
        return elements


class CsvBackend:
    """Structured data as a source type: each CSV row becomes a
    self-contained table element carrying its header context."""

    name = "csv"

    def parse(self, path: Path) -> list[Element]:
        import csv

        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return []
        header = rows[0]
        elements = [
            Element(text=f"Columns: {', '.join(header)}", category="title", page=1)
        ]
        for row in rows[1:]:
            pairs = "; ".join(f"{h}: {v}" for h, v in zip(header, row))
            elements.append(Element(text=pairs, category="table", page=1))
        return elements
