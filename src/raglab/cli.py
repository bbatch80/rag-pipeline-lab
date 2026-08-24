"""raglab command-line interface."""

import re

import click
import psycopg

from raglab import config, db
from raglab.receipts import Receipt


def _redact(url: str) -> str:
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


@click.group()
def main():
    """Governed retrieval pipeline lab."""


@main.command("init-db")
def init_db():
    """Apply db/schema.sql (drop-and-recreate)."""
    receipt = Receipt("raglab init-db")
    try:
        sql = config.SCHEMA_PATH.read_text()
        with db.connect() as conn:
            conn.execute(sql)
        receipt.add("schema", str(config.SCHEMA_PATH))
        receipt.add("tables", "documents, chunks, quarantine (recreated)")
    except (OSError, psycopg.Error) as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("download")
@click.option("--full", is_flag=True, help="Fetch the full corpus, not the dev subset.")
def download(full: bool):
    """Fetch OPM brochures (PDF + BrochureJson listing) into data/raw/."""
    from raglab import corpus, opm

    receipt = Receipt("raglab download" + (" --full" if full else ""))
    counts = {"fetched": 0, "cached": 0, "absent": 0, "failed": 0}
    with opm.client() as http:
        for cell in corpus.cells(dev_only=not full):
            listing = opm.fetch_listing(http, cell.spec.ri, cell.year, cell.listing_path)
            pdf = opm.fetch_pdf(http, cell.spec.ri, cell.year, cell.pdf_path)
            for result in (listing, pdf):
                counts[result.status] += 1
                if result.status == "failed":
                    receipt.fail(f"{cell.spec.ri}/{cell.year}: {result.detail}")
                elif result.status == "absent":
                    receipt.add(f"absent {cell.spec.ri}/{cell.year}", result.detail)
    for status, count in counts.items():
        receipt.add(status, count)
    receipt.finish()


@main.command("ingest")
@click.option("--full", is_flag=True, help="Ingest the full corpus, not the dev subset.")
def ingest_cmd(full: bool):
    """Parse, chunk, gate, and load brochures + internal tier into the database."""
    from raglab import corpus, ingest, internal_corpus
    from raglab.parsing.markdown_backend import CsvBackend, MarkdownBackend
    from raglab.parsing.unstructured_backend import UnstructuredBackend
    from raglab.metadata import derive_document_meta

    receipt = Receipt("raglab ingest" + (" --full" if full else ""))
    backends = {
        "pdf": UnstructuredBackend(),
        "markdown": MarkdownBackend(),
        "csv": CsvBackend(),
    }
    counts = {"skipped": 0, "ingested": 0, "reingested": 0, "quarantined": 0}

    def one(conn, path, meta, backend_kind, label):
        action = ingest.ingest_document(conn, path, meta, backends[backend_kind])
        counts[action] += 1
        if action == "quarantined":
            gates = conn.execute(
                "SELECT gate, detail FROM quarantine WHERE source_path = %s",
                (ingest.rel_source_path(path),),
            ).fetchall()
            tripped = "; ".join(f"{g}: {d}" for g, d in gates)
            receipt.fail(f"QUARANTINED {label} — {tripped}")

    try:
        with db.connect() as conn:
            for cell in corpus.cells(dev_only=not full):
                if not cell.pdf_path.exists():
                    receipt.fail(f"missing PDF (run `raglab download`): {cell.pdf_path.name} {cell.year}")
                    continue
                one(conn, cell.pdf_path, derive_document_meta(cell), "pdf",
                    f"{cell.spec.ri}/{cell.year}")

            internal_items = internal_corpus.items()
            for item in internal_items:
                one(conn, item.path, item.meta, item.backend_kind, item.meta.title)
            if internal_items:
                # Remove rows for internal docs whose source files are gone
                # (churn deletions) — cascade clears their chunks.
                present = [ingest.rel_source_path(i.path) for i in internal_items]
                gone = conn.execute(
                    "DELETE FROM documents WHERE source_path LIKE 'data/internal/%%' "
                    "AND NOT (source_path = ANY(%s)) RETURNING source_path",
                    (present,),
                ).fetchall()
                for (source_path,) in gone:
                    receipt.add("deleted (source gone)", source_path)
            conn.commit()

            for status_name, count in counts.items():
                receipt.add(status_name, count)
            total, histogram = _chunk_histogram(conn)
            receipt.add("chunks total", total)
            receipt.add("size histogram", histogram)
    except psycopg.Error as exc:
        receipt.fail(f"database error: {exc}")
    receipt.finish()


def _chunk_histogram(conn, bucket: int = 250, top: int = 2000) -> tuple[int, str]:
    rows = conn.execute(
        "SELECT width_bucket(length(content), 0, %s, %s) AS b, count(*) "
        "FROM chunks GROUP BY b ORDER BY b",
        (top, top // bucket),
    ).fetchall()
    total = sum(count for _, count in rows)
    bars = " ".join(f"{(b - 1) * bucket}+:{count}" for b, count in rows)
    return total, bars or "empty"


@main.command("embed")
def embed_cmd():
    """Embed all chunks lacking embeddings (resumable; commits per batch)."""
    from openai import OpenAI

    from raglab import embed

    receipt = Receipt("raglab embed")
    try:
        with db.connect() as conn:
            stats = embed.embed_pending(conn, OpenAI())
            remaining = conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NULL"
            ).fetchone()[0]
        receipt.add("embedded", stats.embedded)
        receipt.add("batches", stats.batches)
        receipt.add("tokens", stats.tokens)
        receipt.add("est. cost", f"${stats.cost:.4f}")
        receipt.add("still NULL", remaining)
        if remaining:
            receipt.fail(f"{remaining} chunks still lack embeddings")
    except Exception as exc:  # API errors surface loudly, not as tracebacks
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("index")
def index_cmd():
    """Drop and rebuild the HNSW index (bulk-load-then-index rule)."""
    receipt = Receipt("raglab index")
    try:
        with db.connect() as conn:
            conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
            conn.execute(
                "CREATE INDEX chunks_embedding_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )
            conn.commit()
        receipt.add("index", "chunks_embedding_idx (hnsw, cosine, m=16, ef_construction=64)")
    except psycopg.Error as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("benchmark")
def benchmark_cmd():
    """Benchmark HNSW recall/latency against exact scan across ef_search."""
    from raglab import benchmark

    receipt = Receipt("raglab benchmark")
    try:
        with db.connect() as conn:
            result = benchmark.run(conn)
        click.echo("\n" + benchmark.markdown_table(result) + "\n")
        passing = [r for r in result.rows if r.recall >= 0.95]
        if passing:
            best = min(passing, key=lambda r: r.median_ms)
            receipt.add("operating point", f"ef_search={best.ef_search} "
                        f"(recall {best.recall:.3f}, {best.median_ms:.1f} ms)")
        else:
            receipt.fail("no ef_search value reached recall 0.95")
        receipt.add("exact median", f"{result.exact_median_ms:.1f} ms")
    except psycopg.Error as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("synth")
@click.option("--count", default=250, help="Number of clinical notes.")
@click.option("--seed", default=42, help="Generation seed.")
def synth_cmd(count: int, seed: int):
    """Generate the internal tier: docs, clinical notes + PHI manifest, PDFs."""
    from raglab.synth import internal_docs, notes, render_pdf

    receipt = Receipt("raglab synth")
    try:
        docs_written = internal_docs.write_all()
        receipt.add("internal docs", docs_written)
        with db.connect() as conn:
            stats = notes.generate(conn, count=count, seed=seed)
        receipt.add("clinical notes", stats["notes"])
        receipt.add("by template", stats["by_template"])
        pdfs = render_pdf.render_all()
        receipt.add("rendered PDFs", f"{pdfs} (md+pdf total = notes)")
        manifest_lines = notes.MANIFEST_PATH.read_text().count("\n")
        if manifest_lines != stats["notes"]:
            receipt.fail(f"manifest has {manifest_lines} entries, expected {stats['notes']}")
        receipt.add("PHI manifest", f"{manifest_lines} entries at {notes.MANIFEST_PATH.name}")
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("churn")
@click.option("--seed", required=True, type=int, help="Churn seed (determinism contract).")
@click.option("--rate", default=0.08, help="Fraction of the churnable pool to touch.")
def churn_cmd(seed: int, rate: float):
    """Mutate/delete a slice of the churnable internal docs (golden-anchored spared)."""
    from raglab.synth import churn

    receipt = Receipt(f"raglab churn --seed {seed}")
    try:
        actions = churn.run(seed=seed, rate=rate)
        for action in actions:
            receipt.add(action.action, action.relpath)
        receipt.add("pool size", len(churn.churn_pool()))
        if not actions:
            receipt.fail("churn touched nothing — pool empty?")
    except OSError as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("load-synthea")
def load_synthea_cmd():
    """Load Synthea CSV exports into the synthea schema (drop-and-recreate)."""
    from raglab import synthea_load

    receipt = Receipt("raglab load-synthea")
    try:
        with db.connect() as conn:
            for result in synthea_load.load_all(conn):
                note = (
                    f"{result.rows} rows"
                    + (f" ({result.skipped_csv_columns} csv cols ignored)"
                       if result.skipped_csv_columns else "")
                )
                receipt.add(f"synthea.{result.table}", note)
    except (OSError, psycopg.Error) as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("status")
def status():
    """One-command health snapshot."""
    receipt = Receipt("raglab status")
    try:
        with db.connect() as conn:
            version = conn.execute("SHOW server_version").fetchone()[0]
            receipt.add("database", f"{_redact(config.DATABASE_URL)} (pg {version})")

            vector_ext = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if vector_ext:
                receipt.add("pgvector", vector_ext[0])
            else:
                receipt.fail("pgvector extension not installed")

            for table in ("documents", "chunks", "quarantine"):
                exists = conn.execute(
                    "SELECT to_regclass(%s)", (table,)
                ).fetchone()[0]
                if exists is None:
                    receipt.fail(f"table missing: {table}")
                    continue
                count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                receipt.add(table, count)

            null_embeddings = conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NULL"
            ).fetchone()
            if null_embeddings is not None:
                receipt.add("chunks w/o embedding", null_embeddings[0])

            synthea_patients = conn.execute(
                "SELECT count(*) FROM synthea.patients"
            ).fetchone()[0] if conn.execute(
                "SELECT to_regclass('synthea.patients')"
            ).fetchone()[0] else None
            receipt.add(
                "synthea lane",
                f"{synthea_patients} patients" if synthea_patients is not None
                else "not loaded",
            )

            hnsw = conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_idx'"
            ).fetchone()
            receipt.add("hnsw index", "present" if hnsw else "absent (exact scan)")

            backlog = conn.execute(
                "SELECT source_path, gate FROM quarantine ORDER BY quarantined_at"
            ).fetchall()
            for source_path, gate in backlog:
                receipt.fail(f"quarantined: {source_path} (gate: {gate})")
    except psycopg.Error as exc:
        receipt.fail(f"database unreachable: {exc}")
    receipt.finish()
