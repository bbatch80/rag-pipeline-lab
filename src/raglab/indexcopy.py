"""The search copy of a chunk (chunks.index_text) for normalized sources.

Display copy (chunks.content) is the verbatim de-identified text — what is
cited, shown, and audited. The search copy is what gets embedded and
BM25-indexed. For call notes (D8) it is derived in this order:
  1. identifier normalization — every surface form to its canonical value
  2. abbreviation expansion — the curated shorthand dictionary
  3. boilerplate suppression — sentences that appear in more than
     BOILERPLATE_DF of the source's documents are dropped from the search
     copy (production does not know the macro list; frequency finds it)
The dictionary version and threshold are part of the processing recipe, so
changing either re-ingests exactly the affected source."""

import re
from collections import Counter
from pathlib import Path

from raglab import config, identifiers
from raglab.synth import shorthand

BOILERPLATE_DF = 0.02
RECIPE = f"{shorthand.VERSION}|boiler:{BOILERPLATE_DF}|dedup:v2"
NORMALIZED_SOURCES = {"call_notes"}
_SENTENCES = re.compile(r"(?<=[.;!?])\s+|\n")
_boilerplate_cache: dict[str, frozenset[str]] = {}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCES.split(text) if s.strip()]


def boilerplate_for(source) -> frozenset[str]:
    """Sentences present in > BOILERPLATE_DF of the source's documents on
    disk. Computed once per process per source."""
    if source.key in _boilerplate_cache:
        return _boilerplate_cache[source.key]
    base = config.REPO_ROOT / source.dir
    files = sorted(base.glob("*.md")) if base.exists() else []
    df: Counter = Counter()
    for f in files:  # counted AFTER expansion, the same form suppression sees
        for s in set(sentences(shorthand.expand(f.read_text()))):
            df[s] += 1
    # more than BOILERPLATE_DF of documents AND at least 3 of them (small slices)
    threshold = max(2, BOILERPLATE_DF * len(files))
    found = frozenset(s for s, n in df.items() if n > threshold and len(files) >= 20)
    _boilerplate_cache[source.key] = found
    return found


def normalize(text: str, source) -> str:
    if source.key not in NORMALIZED_SOURCES:
        return text
    out = identifiers.normalize_identifiers(text)
    out = shorthand.expand(out)
    drop = boilerplate_for(source)
    if drop:
        out = "\n".join(s for s in sentences(out) if s not in drop)
    return out


def is_normalized(source) -> bool:
    return source.key in NORMALIZED_SOURCES
