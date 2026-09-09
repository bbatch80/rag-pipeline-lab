"""Heading-aware chunking over normalized elements.

Backend-agnostic on purpose: both parser backends produce the same Element
stream, so chunking behavior is identical and backend comparisons isolate
parse quality.

Headings are soft boundaries: a title closes the current chunk only when
enough content has accumulated (merge_under) — parsers over-detect titles
in dense benefit tables, and hard boundaries there shred the document into
fragments.

Knobs (starting points; chunk size is a Phase 4 experiment):
- hard_max: no chunk exceeds this; oversized elements are split on whitespace
- soft_max: close the current chunk once it reaches this
- merge_under: chunks smaller than this are merged into a neighbor
"""

import os
from dataclasses import dataclass, field

from raglab.parsing.base import Element

# Env-overridable for A/B experiments (chunk size is a measured knob, not a
# tuned-by-feel one). Defaults are the Phase 1 starting points.
HARD_MAX = int(os.environ.get("RAGLAB_CHUNK_HARD", 2000))
SOFT_MAX = int(os.environ.get("RAGLAB_CHUNK_SOFT", 1500))
MERGE_UNDER = int(os.environ.get("RAGLAB_CHUNK_MERGE", 250))


@dataclass
class Chunk:
    text: str
    section: str
    pages: tuple[int, ...] = ()
    categories: tuple[str, ...] = ()


@dataclass
class _Builder:
    section: str
    parts: list[str] = field(default_factory=list)
    pages: set[int] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)

    def size(self) -> int:
        return sum(len(p) for p in self.parts) + 2 * max(0, len(self.parts) - 1)

    def add(self, el: Element) -> None:
        self.parts.append(el.text)
        if el.page is not None:
            self.pages.add(el.page)
        self.categories.add(el.category)

    def build(self) -> Chunk:
        return Chunk(
            text="\n\n".join(self.parts),
            section=self.section,
            pages=tuple(sorted(self.pages)),
            categories=tuple(sorted(self.categories)),
        )


def _split_oversized(el: Element, hard_max: int) -> list[Element]:
    if len(el.text) <= hard_max:
        return [el]
    pieces, remaining = [], el.text
    while len(remaining) > hard_max:
        cut = remaining.rfind(" ", 0, hard_max)
        if cut <= 0:
            cut = hard_max
        pieces.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return [Element(text=p, category=el.category, page=el.page) for p in pieces]


def chunk_elements(
    elements: list[Element],
    hard_max: int = HARD_MAX,
    soft_max: int = SOFT_MAX,
    merge_under: int = MERGE_UNDER,
    profile: str = "section",
) -> list[Chunk]:
    """profile comes from the source row: 'section' (default: split on
    headings, merge small pieces) or 'record' (one document = one chunk —
    call notes, short records — never merged, never split unless it exceeds
    hard_max, in which case it falls back to section chunking)."""
    if profile == "record":
        text = "\n\n".join(el.text for el in elements if el.text.strip()).strip()
        if text and len(text) <= hard_max:
            titles = [el.text for el in elements if el.category == "title"]
            return [Chunk(
                text=text,
                section=titles[0] if titles else "",
                pages=tuple(sorted({el.page for el in elements if el.page is not None})),
                categories=tuple(sorted({el.category for el in elements})),
            )]
    chunks: list[Chunk] = []
    current: _Builder | None = None
    section = ""

    def close() -> None:
        nonlocal current
        if current is not None and current.parts:
            chunks.append(current.build())
        current = None

    for el in elements:
        if el.category == "title":
            section = el.text
            if current is not None and current.size() >= merge_under:
                close()
            if current is None:
                current = _Builder(section=section)
            current.add(el)
            continue
        for piece in _split_oversized(el, hard_max):
            if current is None:
                current = _Builder(section=section)
            elif current.size() + len(piece.text) > soft_max:
                close()
                current = _Builder(section=section)
            current.add(piece)
    close()

    return _merge_small(chunks, merge_under, hard_max)


def _merge_small(chunks: list[Chunk], merge_under: int, hard_max: int) -> list[Chunk]:
    """Fold undersized chunks into the preceding chunk (or the next, for a
    small leading chunk), never growing a chunk past hard_max."""
    out: list[Chunk] = []
    for chunk in chunks:
        if (
            out
            and len(chunk.text) < merge_under
            and len(out[-1].text) + len(chunk.text) + 2 <= hard_max
        ):
            prev = out.pop()
            out.append(
                Chunk(
                    text=prev.text + "\n\n" + chunk.text,
                    section=prev.section,
                    pages=tuple(sorted({*prev.pages, *chunk.pages})),
                    categories=tuple(sorted({*prev.categories, *chunk.categories})),
                )
            )
        elif (
            not out
            and len(chunk.text) < merge_under
            and len(chunks) > 1
        ):
            # Small leading chunk: hold it and let the next chunk absorb it.
            out.append(chunk)
        else:
            out.append(chunk)
    # Second pass for a still-small leading chunk.
    if len(out) >= 2 and len(out[0].text) < merge_under:
        first, second = out[0], out[1]
        if len(first.text) + len(second.text) + 2 <= hard_max:
            out[1] = Chunk(
                text=first.text + "\n\n" + second.text,
                section=second.section or first.section,
                pages=tuple(sorted({*first.pages, *second.pages})),
                categories=tuple(sorted({*first.categories, *second.categories})),
            )
            out = out[1:]
    return out
