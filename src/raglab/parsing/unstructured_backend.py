"""Unstructured backend, `fast` strategy."""

import os
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


class UnstructuredBackend:
    name = "unstructured-fast"

    def parse(self, path: Path) -> list[Element]:
        from unstructured.partition.pdf import partition_pdf

        raw = partition_pdf(filename=str(path), strategy="fast", languages=["eng"])
        elements = []
        for el in raw:
            if el.category in _DROP:
                continue
            text = (el.text or "").strip()
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
