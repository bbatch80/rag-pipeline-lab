"""Normalized parse output. Every backend maps its native element types into
this shape; everything downstream (chunking, metadata, gates) sees only this."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CATEGORIES = ("title", "text", "table", "list_item")


@dataclass(frozen=True)
class Element:
    text: str
    category: str  # one of CATEGORIES
    page: int | None = None


class ParserBackend(Protocol):
    name: str

    def parse(self, path: Path) -> list[Element]: ...
