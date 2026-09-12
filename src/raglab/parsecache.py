"""Parse cache: a parser's elements for a document, stored once and reused
while the inputs are provably unchanged (2026-09-12).

Parsing is deterministic in the PDF bytes, the parser, and the parser's
library version; hi_res costs ~11 minutes per brochure, chunking costs
milliseconds. Keeping the parse lets a chunking change re-chunk in seconds
instead of re-parsing for hours. The key holds everything the parse depends
on, so a stale entry can never match: source bytes (sha256), backend name,
library versions. RAGLAB_PARSE_CACHE=off forces a fresh parse (the proof
that the cache is still honest). Files live under data/, never in git.
"""

import hashlib
import json
import os
from pathlib import Path

from raglab import config
from raglab.parsing.base import Element

CACHE_DIR = config.REPO_ROOT / "data" / "parse_cache"


def _library_versions(backend_name: str) -> str:
    import importlib.metadata as m

    names = ("unstructured", "unstructured-inference") if backend_name.startswith("unstructured") else ()
    return ",".join(f"{n}={m.version(n)}" for n in names)


def cache_key(path: Path, backend_name: str) -> str:
    digest = hashlib.sha256(path.read_bytes())
    digest.update(f"|{backend_name}|{_library_versions(backend_name)}".encode())
    return digest.hexdigest()


def enabled() -> bool:
    return os.environ.get("RAGLAB_PARSE_CACHE", "on") != "off"


def load(path: Path, backend_name: str) -> list[Element] | None:
    if not enabled():
        return None
    entry = CACHE_DIR / f"{cache_key(path, backend_name)}.json"
    if not entry.exists():
        return None
    data = json.loads(entry.read_text())
    return [Element(text=e["text"], category=e["category"], page=e.get("page")) for e in data["elements"]]


def store(path: Path, backend_name: str, elements: list[Element]) -> None:
    if not enabled():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = CACHE_DIR / f"{cache_key(path, backend_name)}.json"
    entry.write_text(json.dumps({
        "source": str(path), "backend": backend_name, "versions": _library_versions(backend_name),
        "elements": [{"text": e.text, "category": e.category, "page": e.page} for e in elements],
    }))


def parse(backend, path: Path) -> tuple[list[Element], bool]:
    """(elements, from_cache): the cached parse when its key matches, else a
    fresh parse that is then stored."""
    cached = load(path, backend.name)
    if cached is not None:
        return cached, True
    elements = backend.parse(path)
    store(path, backend.name, elements)
    return elements, False
