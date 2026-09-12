"""Docling backend. Model weights are pre-fetched by `docling-tools models
download` (an explicit setup step); artifacts_path pins the parser to those
local weights so nothing downloads at runtime."""

from pathlib import Path

from raglab.parsing.base import Element

MODELS_DIR = Path.home() / ".cache" / "docling" / "models"

_DROP_LABELS = {"page_header", "page_footer"}

_CATEGORY_MAP = {
    "title": "title",
    "section_header": "title",
    "table": "table",
    "list_item": "list_item",
}


class DoclingBackend:
    name = "docling"

    def parse(self, path: Path) -> list[Element]:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if not MODELS_DIR.exists():
            raise RuntimeError(
                "Docling model weights missing — run `docling-tools models download`"
            )
        options = PdfPipelineOptions(artifacts_path=MODELS_DIR)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        doc = converter.convert(str(path)).document

        elements = []
        for item, _level in doc.iterate_items():
            label = str(getattr(item, "label", ""))
            if label in _DROP_LABELS:
                continue
            if label == "table":
                text = item.export_to_markdown(doc=doc).strip()
            else:
                text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            page = item.prov[0].page_no if getattr(item, "prov", None) else None
            elements.append(
                Element(
                    text=text,
                    category=_CATEGORY_MAP.get(label, "text"),
                    page=page,
                )
            )
        return elements
