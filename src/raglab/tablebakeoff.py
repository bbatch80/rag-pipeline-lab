"""Parser bake-off on table structure (Phase 3.5, 2026-09-12).

Parsers read the same brochures into a scratch table, side by side (fast,
hi_res; Docling took part in the 2026-09-12 bake-off and was removed after
hi_res was promoted — its scratch rows remain as the record).
The measurement is whether a TABLE ROW survives parsing and chunking: for
a labeled cell, does one chunk hold the value, its row label, and its
column header together? hit@5 never asked that (any page stating the value
counts), which is how Phase 0's bake-off could pick a parser that leaves
zero table elements in 11,022 brochure chunks.

Labels live in eval/table_cells.jsonl: {plan_code, year, printed_page,
value, row, column, from}. Nothing here touches the live corpus.
"""

import json
import re
import time
from pathlib import Path
from statistics import median

import psycopg

from raglab import config, corpus
from raglab.chunking import chunk_elements

CELLS_PATH = config.REPO_ROOT / "eval" / "table_cells.jsonl"
EVAL_SCHEMA_PATH = config.REPO_ROOT / "db" / "eval.sql"

PARSERS = ("fast", "hires")  # docling lost the promotion (2026-09-12) and was removed


def backend_for(parser: str):
    from raglab.parsing.unstructured_backend import UnstructuredBackend
    return UnstructuredBackend("fast" if parser == "fast" else "hi_res")


def brochure_cells(years: tuple[int, ...] = ()):
    return [c for c in corpus.cells(dev_only=False) if not years or c.year in years]


def parse_into_scratch(conn: psycopg.Connection, parser: str, years: tuple[int, ...] = (),
                       log=print) -> dict:
    """Parse every brochure (or the given years) with one parser and store
    its chunks under that parser's name, replacing any earlier run."""
    conn.execute(EVAL_SCHEMA_PATH.read_text())
    backend = backend_for(parser)
    stats = {"parser": backend.name, "documents": 0, "chunks": 0, "table_chunks": 0, "seconds": 0.0}
    for cell in brochure_cells(years):
        rel = str(cell.pdf_path.relative_to(config.REPO_ROOT))
        t0 = time.perf_counter()
        elements = backend.parse(cell.pdf_path)
        chunks = chunk_elements(elements, profile="section")
        ms = (time.perf_counter() - t0) * 1000
        with conn.transaction():
            conn.execute("DELETE FROM chunks_bakeoff WHERE parser = %s AND source_path = %s", (backend.name, rel))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks_bakeoff (parser, source_path, chunk_index, section, pages, categories, content, parse_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [(backend.name, rel, i, c.section, list(c.pages), list(c.categories), c.text, ms)
                     for i, c in enumerate(chunks)],
                )
        conn.commit()
        tables = sum(1 for c in chunks if "table" in c.categories)
        stats["documents"] += 1; stats["chunks"] += len(chunks); stats["table_chunks"] += tables; stats["seconds"] += ms / 1000
        log(f"{backend.name}: {rel} -> {len(chunks)} chunks ({tables} table) in {ms/1000:.0f}s")
    return stats


def load_cells() -> list[dict]:
    if not CELLS_PATH.exists():
        return []
    return [json.loads(line) for line in CELLS_PATH.read_text().splitlines() if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _table_rows(text: str) -> list[tuple[list[str], str]]:
    """(header cells, row line) for every body row of every markdown table
    in the text; the header is the first row above the separator."""
    out, header = [], None
    for line in text.split("\n"):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            header = None
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells) and cells:
            continue  # the separator: the previous row was the header
        if header is None:
            header = cells
            continue
        out.append((header, stripped))
    return out


def row_intact(chunks: list[str], cell: dict) -> bool:
    """True when the value and the row label sit on ONE table row whose
    header names the column — the row survived as a row. Co-occurrence of
    the three strings anywhere in a chunk is not enough (a prose paragraph
    mentioning Elevate, specialist and $30 in three sentences scored as a
    'row' under that definition)."""
    value, row, column = _norm(cell["value"]), _norm(cell["row"]), _norm(cell["column"])
    for text in chunks:
        for header, line in _table_rows(text):
            if value in _norm(line) and row in _norm(line) and any(column in _norm(h) for h in header):
                return True
            # a row-label column: the row label may be the first cell and the header cell the value's column
        # A plain-text row (no markdown) counts only when one line holds all
        # three AND reads as a row: cells separated by pipes or tabs, not prose.
        for line in text.split("\n"):
            n = _norm(line)
            if ("|" in line or "\t" in line) and value in n and row in n and column in n:
                return True
    return False


def report(conn: psycopg.Connection) -> dict:
    """Per parser: structure statistics and the table-row metric over the
    labeled cells. Cells whose document a parser has not parsed are skipped
    for that parser and counted as 'unparsed'."""
    conn.execute(EVAL_SCHEMA_PATH.read_text())
    out = {}
    parsers = [r[0] for r in conn.execute("SELECT DISTINCT parser FROM chunks_bakeoff ORDER BY 1").fetchall()]
    cells = load_cells()
    for parser in parsers:
        rows = conn.execute(
            "SELECT source_path, section, categories, content FROM chunks_bakeoff WHERE parser = %s", (parser,)
        ).fetchall()
        by_doc: dict[str, list[str]] = {}
        for src, _sec, _cats, content in rows:
            by_doc.setdefault(src, []).append(content)
        sizes = [len(r[3]) for r in rows]
        cell_headings = sum(1 for r in rows if re.fullmatch(r"(all charges|high option|standard option|elevate( plus)?|\$[\d,.]+.*|in-network.*|nothing.*)", (r[1] or "").strip().lower()))
        intact, missing, unparsed, detail = 0, [], 0, []
        for cell in cells:
            src = f"data/raw/{cell['year']}/{cell['plan_code']}.pdf"
            if src not in by_doc:
                unparsed += 1; continue
            ok = row_intact(by_doc[src], cell)
            intact += ok
            detail.append((cell.get("from", "?"), ok))
            if not ok:
                missing.append(cell.get("from", "?"))
        out[parser] = {
            "documents": len(by_doc), "chunks": len(rows),
            "table_chunks": sum(1 for r in rows if "table" in (r[2] or [])),
            "median_chars": int(median(sizes)) if sizes else 0,
            "cell_heading_chunks": cell_headings,
            "parse_s_per_doc": round((conn.execute("SELECT avg(parse_ms) FROM chunks_bakeoff WHERE parser = %s", (parser,)).fetchone()[0] or 0) / 1000, 1),
            "cells_scored": len(cells) - unparsed, "rows_intact": intact,
            "row_intact_rate": round(intact / (len(cells) - unparsed), 3) if len(cells) - unparsed else None,
            "rows_missing": missing,
        }
    return out
