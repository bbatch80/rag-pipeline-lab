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
    """Parse, chunk, gate, and load brochures into the database."""
    from raglab import corpus, ingest
    from raglab.parsing.unstructured_backend import UnstructuredBackend

    receipt = Receipt("raglab ingest" + (" --full" if full else ""))
    backend = UnstructuredBackend()
    counts = {"skipped": 0, "ingested": 0, "reingested": 0, "quarantined": 0}
    try:
        with db.connect() as conn:
            for cell in corpus.cells(dev_only=not full):
                if not cell.pdf_path.exists():
                    receipt.fail(f"missing PDF (run `raglab download`): {cell.pdf_path.name} {cell.year}")
                    continue
                from raglab.metadata import derive_document_meta

                action = ingest.ingest_document(
                    conn, cell.pdf_path, derive_document_meta(cell), backend
                )
                counts[action] += 1
                if action == "quarantined":
                    gates = conn.execute(
                        "SELECT gate, detail FROM quarantine WHERE source_path = %s",
                        (ingest.rel_source_path(cell.pdf_path),),
                    ).fetchall()
                    tripped = "; ".join(f"{g}: {d}" for g, d in gates)
                    receipt.fail(f"QUARANTINED {cell.spec.ri}/{cell.year} — {tripped}")
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

            backlog = conn.execute(
                "SELECT source_path, gate FROM quarantine ORDER BY quarantined_at"
            ).fetchall()
            for source_path, gate in backlog:
                receipt.fail(f"quarantined: {source_path} (gate: {gate})")
    except psycopg.Error as exc:
        receipt.fail(f"database unreachable: {exc}")
    receipt.finish()
