"""Internal-tier documents for ingest: file -> (metadata, backend).

Every vector-lane source under data/internal is discovered from its
`sources.dir` — code never names a source's directory. Convention: a
markdown-authored source may also ship a PDF rendition in `<dir>_pdf`
(clinical notes do). Authored docs keep their registered titles; generated
records use the file stem. Member-scoped sources carry the person key from
their PHI manifest."""

import json
from dataclasses import dataclass
from pathlib import Path

from raglab import config
from raglab.metadata import DocumentMeta, derive_internal_meta
from raglab.sources import Registry, Source
from raglab.synth.internal_docs import ALL_DOCS, INTERNAL_DIR

MANIFEST_DIR = INTERNAL_DIR / "manifests"
_EXT = {"markdown": "*.md", "csv": "*.csv", "pdf": "*.pdf"}


@dataclass(frozen=True)
class InternalItem:
    path: Path
    meta: DocumentMeta
    backend_kind: str  # markdown | csv | pdf


def manifest_meta(source: Source) -> dict[str, dict]:
    """file stem -> {patient_id, year} from the source's manifest (if it has
    one). `year` is the year of the event the document records (a call's
    date), not the ingest year."""
    path = MANIFEST_DIR / f"{source.key}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[Path(rec["doc"]).stem] = {"patient_id": rec.get("patient_id"), "year": rec.get("year")}
    return out


def member_keys(source: Source) -> dict[str, str]:
    """file stem -> person key, from the source's manifest (if it has one)."""
    return {stem: m["patient_id"] for stem, m in manifest_meta(source).items() if m["patient_id"]}


def files_for(source: Source) -> list[tuple[Path, str]]:
    """(path, backend kind) for every file of a discoverable source."""
    if not source.dir or not source.dir.startswith("data/internal/") or not source.parser:
        return []
    base = config.REPO_ROOT / source.dir
    found = []
    if base.exists() and source.parser in _EXT:
        found += [(p, source.parser) for p in sorted(base.glob(_EXT[source.parser]))]
    pdf_dir = base.with_name(base.name + "_pdf")
    if source.parser == "markdown" and pdf_dir.exists():
        found += [(p, "pdf") for p in sorted(pdf_dir.glob("*.pdf"))]
    return found


def items(registry: Registry) -> list[InternalItem]:
    titles = {doc.relpath: doc.title for doc in ALL_DOCS}
    out = []
    for source in registry.all:
        if source.lane not in ("vector", "both") or source.status != "ingested":
            continue
        metas = manifest_meta(source)
        for path, kind in files_for(source):
            rel = str(path.relative_to(INTERNAL_DIR))
            m = metas.get(path.stem, {})
            out.append(InternalItem(
                path=path,
                meta=derive_internal_meta(
                    titles.get(rel, path.stem), source.doc_type, source.acl_tag,
                    year=m.get("year") or 2026,
                    member_key=m.get("patient_id"),
                ),
                backend_kind=kind,
            ))
    return out
