"""raglab command-line interface."""

import json
import os
import re
import subprocess

from pathlib import Path

import click
import psycopg

from raglab import config, db
from raglab.receipts import Receipt


def _clip(text: str, limit: int = 58) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\u2026"


def _redact(url: str) -> str:
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


@click.group()
def main():
    """Governed retrieval pipeline lab."""


@main.command("init-db")
def init_db():
    """Fresh database: db/schema.sql (drop-and-recreate), governance, then
    every migration. Existing databases use `raglab migrate` instead."""
    from raglab import migrations

    receipt = Receipt("raglab init-db")
    try:
        with db.connect() as conn:
            conn.execute(config.SCHEMA_PATH.read_text())
            conn.execute(config.GOVERNANCE_PATH.read_text())
            conn.execute((config.REPO_ROOT / "db" / "eval.sql").read_text())  # additive
            conn.commit()
            conn.execute("DELETE FROM schema_migrations") if _table_exists(conn, "schema_migrations") else None
            conn.execute("DROP TABLE IF EXISTS sources CASCADE")
            conn.commit()
            ran = migrations.apply(conn)
        receipt.add("schema", str(config.SCHEMA_PATH))
        receipt.add("tables", "documents, chunks, quarantine (recreated)")
        receipt.add("governance", "RLS policies + personas + disclosure_log applied")
        receipt.add("eval store", "eval_runs, eval_scores (additive)")
        receipt.add("migrations", ", ".join(f"{m.version:03d}_{m.name}" for m in ran) or "none")
    except (OSError, psycopg.Error, ValueError) as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


def _table_exists(conn, name: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()[0]


@main.command("migrate")
def migrate_cmd():
    """Apply pending db/migrations/*.sql, in order, once each. The corpus
    stays put: every migration is additive."""
    from raglab import migrations

    receipt = Receipt("raglab migrate")
    try:
        with db.connect() as conn:
            done_before = migrations.applied(conn)
            ran = migrations.apply(conn)
            pending = [m for m in migrations.available() if m.version not in done_before]
        for m in ran:
            receipt.add("applied", f"{m.version:03d}_{m.name}")
        if not ran:
            receipt.add("applied", "nothing pending")
        receipt.add("schema version", max((m.version for m in migrations.available()), default=0))
        assert len(pending) == len(ran)
    except (OSError, psycopg.Error, ValueError) as exc:
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
    from raglab import corpus, ingest, internal_corpus, sources
    from raglab.parsing.markdown_backend import CsvBackend, MarkdownBackend
    from raglab.parsing.unstructured_backend import UnstructuredBackend
    from raglab.metadata import derive_document_meta

    receipt = Receipt("raglab ingest" + (" --full" if full else ""))
    backends = {
        "pdf": UnstructuredBackend(),
        "markdown": MarkdownBackend(),
        "csv": CsvBackend(),
    }
    counts = {"skipped": 0, "ingested": 0, "reingested": 0, "quarantined": 0, "duplicate": 0}

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

            registry = sources.load(conn)
            from raglab import dedup

            for src in registry.all:  # originals already ingested must be recognizable
                if src.chunk_profile == "record" and src.status == "ingested":
                    dedup.seed(src.key, conn)
            internal_items = internal_corpus.items(registry)
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


# Pinned explicitly: pgvector's defaults today, but an index rebuilt on
# another machine or version must produce the same graph parameters.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


@main.command("index")
def index_cmd():
    """Drop and rebuild the HNSW index; reindex BM25 (bulk-load-then-index)."""
    receipt = Receipt("raglab index")
    try:
        # Dead row versions left by bulk updates are counted by index builds
        # (BM25 statistics, HNSW graph) — vacuum first, outside a transaction.
        with db.connect() as vconn:
            vconn.autocommit = True
            vconn.execute("VACUUM ANALYZE chunks")
        with db.connect() as conn:
            conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
            conn.execute(
                "CREATE INDEX chunks_embedding_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
            # BM25 (pg_textsearch) is fastest built after a bulk load too —
            # one index per source (own statistics); reindex each.
            bm25 = [r[0] for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'chunks_bm25_%%' ORDER BY 1"
            ).fetchall()]
            for name in bm25:
                conn.execute(f"REINDEX INDEX {name}")
            conn.commit()
        receipt.add("vacuum", "chunks (dead row versions cleared before index builds)")
        receipt.add("index", f"chunks_embedding_idx (hnsw, cosine, m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})")
        receipt.add("bm25", f"{len(bm25)} per-source indexes reindexed (pg_textsearch, english)")
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


@main.group("synth")
def synth_group():
    """Generate synthetic sources, one command per source (each writes its
    own PHI manifest under data/internal/manifests/)."""


@synth_group.command("docs")
def synth_docs_cmd():
    """Authored internal docs: SOPs, bulletins, formulary, KB, rates."""
    from raglab.synth import internal_docs

    receipt = Receipt("raglab synth docs")
    try:
        receipt.add("internal docs", internal_docs.write_all())
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@synth_group.command("calls")
@click.option("--members", default=1500, help="Members with call histories (floor).")
@click.option("--calls", default=10000, help="Target number of call notes.")
@click.option("--seed", default=42, help="Generation seed.")
def synth_calls_cmd(members: int, calls: int, seed: int):
    """Call notes from per-member storylines (+ synthea.call_log rows) and their PHI manifest."""
    from raglab.synth import journeys

    receipt = Receipt("raglab synth calls")
    try:
        with db.connect() as conn:
            stats = journeys.generate(conn, members=members, target_calls=calls, seed=seed)
            conn.commit()
        for k, v in stats.items():
            receipt.add(k, v)
        receipt.add("PHI manifest", str(journeys.MANIFEST_PATH.relative_to(config.REPO_ROOT)))
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@synth_group.command("notes")
@click.option("--count", default=250, help="Number of clinical notes.")
@click.option("--seed", default=42, help="Generation seed.")
def synth_notes_cmd(count: int, seed: int):
    """Clinical notes (markdown + PDF renditions) and their PHI manifest."""
    from raglab.synth import notes, render_pdf

    receipt = Receipt("raglab synth notes")
    try:
        with db.connect() as conn:
            stats = notes.generate(conn, count=count, seed=seed)
        receipt.add("clinical notes", stats["notes"])
        receipt.add("by template", stats["by_template"])
        pdfs = render_pdf.render_all()
        receipt.add("rendered PDFs", f"{pdfs} (md+pdf total = notes)")
        manifest_lines = notes.MANIFEST_PATH.read_text().count("\n")
        if manifest_lines != stats["notes"]:
            receipt.fail(f"manifest has {manifest_lines} entries, expected {stats['notes']}")
        receipt.add("PHI manifest", f"{manifest_lines} entries at {notes.MANIFEST_PATH.relative_to(config.REPO_ROOT)}")
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("churn")
@click.option("--seed", required=True, type=int, help="Churn seed (determinism contract).")
@click.option("--rate", default=0.08, help="Fraction of the churnable pool to touch.")
def churn_cmd(seed: int, rate: float):
    """Mutate/delete a slice of the churnable internal docs (golden-anchored spared)."""
    from raglab import sources
    from raglab.synth import churn
    from raglab.synth.internal_docs import INTERNAL_DIR

    receipt = Receipt(f"raglab churn --seed {seed}")
    try:
        with db.connect() as conn:
            globs = churn.pool_globs_for(sources.load(conn))
        actions = churn.run(seed=seed, rate=rate, base_dir=INTERNAL_DIR, pool_globs=globs)
        for action in actions:
            receipt.add(action.action, action.relpath)
        receipt.add("pool size", len(churn.churn_pool(INTERNAL_DIR, globs)))
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


@main.command("deid-eval")
@click.option("--sample", default=None, type=int, help="Seeded sample of documents per source (CI).")
@click.option("--gate", is_flag=True, help="Fail below the D9 thresholds (structured recall, leakage) per source.")
def deid_eval_cmd(sample: int | None, gate: bool):
    """Type-correct PHI detection recall and leakage, per source, against each
    source's PHI manifest (ground truth by construction)."""
    from raglab import deid

    receipt = Receipt("raglab deid-eval" + (f" --sample {sample}" if sample else "") + (" --gate" if gate else ""))
    try:
        with db.connect() as conn:
            result = deid.evaluate(conn, sample=sample, gate=gate)
        receipt.add("run id", result.run_id)
        for key, r in result.by_source.items():
            receipt.add(f"[{key}]", f"{r['n_docs']} docs, {r['n_entities']} entities")
            for entity_type, recall in r["recall_by_type"].items():
                flag = " *" if entity_type in deid.STRUCTURED else ""
                receipt.add(f"  recall {entity_type}{flag}", recall)
            receipt.add("  overall recall", r["overall_recall"])
            receipt.add("  LEAKAGE RATE", f"{r['leakage_rate']:.4f} (surface or canonical value surviving in indexed text)")
            receipt.add("  over-redaction", f"{r['over_redaction_rate']:.4f} of {r['n_applied']} replacements protect nothing (not gated)")
            for etype, span, n in r["over_examples"][:5]:
                receipt.add("    over-redacted", f"{etype} {span!r} ×{n}")
            for src, doc, etype, value in r["examples"][:3]:
                receipt.add("    leaked", f"{doc}: {etype} {value!r}")
        receipt.add("gate", f"structured recall >= {deid.DEID_THRESHOLDS['structured_recall']} (*), leakage < {deid.DEID_THRESHOLDS['leakage_rate']}, per source")
        for failure in result.failures:
            receipt.fail(f"THRESHOLD: {failure}")
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("explain-golden")
@click.argument("qid")
def explain_cmd(qid: str):
    """Why a golden question hits or misses: pool membership, reranker
    position and score, and the text the reranker scored."""
    from raglab import explain as explain_mod

    receipt = Receipt(f"raglab explain-golden {qid}")
    try:
        with db.connect() as conn:
            ex = explain_mod.explain(conn, qid)
        receipt.add("question", ex.question)
        if ex.translated != ex.question:
            receipt.add("translated", ex.translated)
        receipt.add("member context", ex.member_key or "none")
        receipt.add("route", f"{ex.route['scope']} years={ex.route['years']} plans={ex.route['plan_codes']} sources={ex.route['sources'] or 'all'}")
        receipt.add("pool", f"{ex.pool_size} candidates  " + "  ".join(f"{k}={v}" for k, v in sorted(ex.pool_by_source.items())))
        for e in ex.expected:
            name = e["spec"].get("internal") or e["spec"].get("title") or str(e["spec"])
            if not e["in_pool"]:
                receipt.add(f"expected {name}", "NOT IN POOL (retrieval miss: neither arm surfaced it)")
                continue
            receipt.add(f"expected {name}", f"pool rank {e['pool_rank']}, reranked to position {e['rerank_position']} (score {e['score']:.3f})")
            receipt.add("  scored text", (e["scored_text"] or "").replace("\n", " | ")[:300])
        abstained, best = ex.verdict
        receipt.add("verdict", f"{'ABSTAIN' if abstained else 'answer'} (best {best:.3f}, threshold {rerank_threshold()})")
        for i, t in enumerate(ex.top[:5]):
            receipt.add(f"  top {i}", f"{t['title']}  {t['score']:.3f}  {t['text'][:90].replace(chr(10), ' ')}")
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


def rerank_threshold() -> float:
    from raglab import rerank

    return rerank.ABSTAIN_THRESHOLD


@main.command("audit")
@click.option("--document", default=None, help="Which payloads used this doc (title/path substring)?")
@click.option("--persona", default=None, help="What did this persona see?")
@click.option("--limit", default=10)
def audit_cmd(document: str | None, persona: str | None, limit: int):
    """Disclosure-log reports: who saw what; lineage in both directions."""
    receipt = Receipt("raglab audit")
    try:
        with db.connect() as conn:
            if document:
                rows = conn.execute(
                    "SELECT asked_at, persona, query, payload_id FROM disclosure_log "
                    "WHERE EXISTS (SELECT 1 FROM unnest(doc_titles) t WHERE t ILIKE %s) "
                    "ORDER BY id DESC LIMIT %s",
                    (f"%{document}%", limit),
                ).fetchall()
                receipt.add("question", f"which payloads used documents matching {document!r}")
                for asked, who, query, pid in rows:
                    receipt.add(f"  {asked:%m-%d %H:%M}", f"{who:10} {_clip(query)}")
                    receipt.add("    payload_id", str(pid))
                receipt.add("matches", len(rows))
            elif persona:
                rows = conn.execute(
                    "SELECT asked_at, payload_status, query, acl_basis, payload_id "
                    "FROM disclosure_log "
                    "WHERE persona = %s ORDER BY id DESC LIMIT %s",
                    (persona, limit),
                ).fetchall()
                receipt.add("question", f"what did persona {persona!r} see")
                for asked, status, query, basis, pid in rows:
                    receipt.add(f"  {asked:%m-%d %H:%M}",
                                f"{status:22} tiers={','.join(basis)}  {_clip(query)}")
                    receipt.add("    payload_id", str(pid))
                receipt.add("matches", len(rows))
            else:
                total, personas = conn.execute(
                    "SELECT count(*), count(DISTINCT persona) FROM disclosure_log"
                ).fetchone()
                receipt.add("disclosures", total)
                receipt.add("personas", personas)
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("explain")
@click.argument("query")
@click.option("--persona", default=None, type=click.Choice(["public", "employee", "care_team"]))
@click.option("--generate", is_flag=True,
              help="Also send the payload to both generators and print their answers.")
def explain_cmd(query: str, persona: str | None, generate: bool):
    """Full retrieval trace: router -> per-method -> RRF -> rerank -> verdict."""
    from raglab import rerank, retrieval, router

    decision = router.route(query)
    click.echo(f"\nQUERY: {query}")
    click.echo("\n[1] ROUTER")
    click.echo(f"    scope: {decision.scope}")
    for reason in decision.reasons:
        click.echo(f"    - {reason}")
    if decision.scope != "in_scope":
        click.echo(f"    boundary response: {decision.boundary_response}")
        click.echo("\n    (no retrieval attempted — scope gate)")
        return
    click.echo(f"    year filter: {list(decision.years)}")
    click.echo(f"    plan filter: {list(decision.plan_codes) or 'none'} (NULL plan_code passes)")

    search_query = query
    with db.connect() as conn:
        # Mirror the pipeline exactly: vault-entitled sessions (admin,
        # care_team) get query translation via the owner connection.
        if persona in (None, "care_team"):
            from raglab import deid

            search_query = deid.translate_query(conn, query)
            if search_query != query:
                click.echo("\n[1b] VAULT TRANSLATION (re-identification entitlement)")
                click.echo(f"    search query: {search_query}")
        if persona:
            conn.execute(f"SET LOCAL ROLE persona_{persona}")
            click.echo(f"    RLS: querying as persona_{persona} (engine trims before ranking)")
        candidates = retrieval.search(
            conn, search_query, retrieval.embed_query(search_query), decision
        )
        if persona:
            conn.execute("RESET ROLE")

    def _line(c, extra=""):
        loc = f"{c.doc_title[:38]} | {c.section[:24]}" if c.section else c.doc_title[:64]
        return f"    {extra}[{loc}] {c.content[:64].replace(chr(10), ' ')}"

    by_vec = sorted((c for c in candidates if c.vector_rank), key=lambda c: c.vector_rank)
    by_txt = sorted((c for c in candidates if c.text_rank), key=lambda c: c.text_rank)
    click.echo(f"\n[2] VECTOR top 3 (of {len(by_vec)} in fused set)")
    for c in by_vec[:3]:
        click.echo(_line(c, f"v#{c.vector_rank} "))
    click.echo(f"\n[3] BM25 top 3 (of {len(by_txt)} in fused set)")
    for c in by_txt[:3]:
        click.echo(_line(c, f"t#{c.text_rank} "))

    by_source: dict = {}
    for c in candidates:
        by_source.setdefault(c.doc_type, [0, 0])
        by_source[c.doc_type][0] += 1
        by_source[c.doc_type][1] += int(c.floor)
    click.echo(f"\n[3b] POOL BY SOURCE (floor={retrieval.SOURCE_FLOOR}): " + ", ".join(
        f"{t}={n}" + (f" (+{f} floor)" if f else "") for t, (n, f) in sorted(by_source.items())))
    click.echo(f"\n[4] RRF FUSION (k={retrieval.RRF_K}) top 5 of {len(candidates)}")
    for c in sorted(candidates, key=lambda x: -x.rrf_score)[:5]:
        v = f"1/(60+{c.vector_rank})" if c.vector_rank else "0"
        t = f"1/(60+{c.text_rank})" if c.text_rank else "0"
        click.echo(_line(c, f"{c.rrf_score:.4f} = {v} + {t}  "))

    reranked = rerank.rerank(search_query, candidates, stratify_years=decision.years)
    click.echo(f"\n[5] RERANK ({rerank.RERANKER}: {rerank.MODEL_NAME}) top 5"
               + (" — year-stratified" if len(decision.years) > 1 else ""))
    for c in reranked[:5]:
        click.echo(_line(c, f"{c.rerank_score:.4f}  "))

    abstain, best = rerank.abstention_verdict(reranked)
    click.echo("\n[6] VERDICT")
    click.echo(f"    best rerank score: {best:.4f} vs threshold {rerank.ABSTAIN_THRESHOLD}")
    click.echo(f"    {'ABSTAIN (insufficient evidence)' if abstain else 'ANSWERABLE'}")
    click.echo(
        f"    RLS: {'persona_' + persona + ' — invisible tiers never entered retrieval' if persona else 'admin view (all tiers)'}\n"
    )

    if generate:
        from raglab import payload as payload_mod
        from raglab.generators import GENERATORS

        built = payload_mod.build(query, decision, reranked)
        built["persona"] = persona or "admin"
        click.echo("[7] GENERATION (identical payload to both models)")
        click.echo(f"    payload: status={built['status']} "
                   f"confidence={built.get('confidence')} "
                   f"chunks={len(built['chunks'])}")
        for generator in GENERATORS:
            click.echo(f"\n    ──── {generator.name} ────")
            for line in generator.generate(built).splitlines():
                click.echo(f"    {line}")
        click.echo()


@main.command("ablation")
def ablation_cmd():
    """Run the retrieval ablation over the golden set."""
    from raglab import ablation

    receipt = Receipt("raglab ablation")
    try:
        with db.connect() as conn:
            report = ablation.run(conn)
        click.echo("\n" + ablation.markdown_table(report) + "\n")
        for arm, result in report.arms.items():
            receipt.add(arm, f"{result.rate:.3f}")
        gates_ok = sum(1 for _, exp, act in report.gate_results if exp == act)
        receipt.add("scope gate", f"{gates_ok}/{len(report.gate_results)} as expected")
        if report.answerable_best_scores:
            lo = min(s for _, s in report.answerable_best_scores)
            receipt.add("answerable best-score min", f"{lo:.4f}")
        for qid, score in report.unanswerable_best_scores:
            receipt.add(f"unanswerable {qid} best-score", f"{score:.4f}")
        misses = [
            f"{r['id']}:{','.join(a for a in ('vector','bm25','rrf','rrf+rerank') if not r[a])}"
            for r in report.per_question
            if not all(r[a] for a in ("vector", "lexical", "rrf", "rrf+rerank"))
        ]
        for miss in misses:
            receipt.add("miss", miss)
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("eval-retrieval")
@click.option("--label", default="baseline", help="config_label recorded with the run.")
@click.option("--gate", is_flag=True, help="Exit non-zero if thresholds are breached.")
@click.option("--sabotage", is_flag=True, help="Discrimination check: junk query vectors.")
@click.option("--category", "categories", multiple=True,
              help="Only these golden categories (iteration aid; partial runs never gate).")
def eval_retrieval_cmd(label: str, gate: bool, sabotage: bool, categories: tuple[str, ...]):
    """Tier-1 deterministic retrieval eval over the golden set (free)."""
    if categories and gate:
        raise click.UsageError("--gate needs the whole golden set; drop --category")
    from raglab import eval_retrieval
    from raglab.timing import BUDGET_P95_MS

    receipt = Receipt("raglab eval-retrieval" + (" --sabotage" if sabotage else ""))
    try:
        with db.connect() as conn:
            result = eval_retrieval.run(conn, config_label=label, sabotage=sabotage,
                                        categories=tuple(categories))
        receipt.add("run id", result.run_id)
        receipt.add("corpus", result.corpus_hash[:12])
        for metric, value in result.overall.items():
            if value is not None:
                ci = result.overall_ci.get(metric)
                receipt.add(metric, f"{value:.3f}" + (f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        for category, metrics in sorted(result.by_category.items()):
            receipt.add(f"  {category}", _fmt_slice(metrics))
        for slice_name, metrics in sorted(result.by_source.items()):
            receipt.add(f"  source:{slice_name}", _fmt_slice(metrics))
        for stage, pct in result.latency.items():
            budget = f"  (budget p95 ≤ {BUDGET_P95_MS} ms, displayed not gated)" if stage == "total" else ""
            receipt.add(f"latency {stage}", f"p50 {pct['p50']:.0f} ms  p95 {pct['p95']:.0f} ms{budget}")
        if result.diff_against is not None:
            if result.diff:
                for metric, d in result.diff.items():
                    receipt.add(f"vs run {result.diff_against}: {metric}",
                                _fmt_diff(d))
            else:
                receipt.add(f"vs run {result.diff_against}", "no question flipped")
        for failure in result.failures:
            if gate and not sabotage:
                receipt.fail(f"THRESHOLD: {failure}")
            else:
                receipt.add("below threshold", failure)
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


def _fmt_slice(metrics: dict) -> str:
    """hit@5=0.9 [0.70, 0.97]  precision@5=0.32 ...  n=20"""
    parts = []
    for m, v in metrics.items():
        if m.endswith("_ci") or m == "n":
            continue
        ci = metrics.get(f"{m}_ci")
        parts.append(f"{m}={v}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
    parts.append(f"n={metrics.get('n', '?')}")
    return "  ".join(parts)


def _fmt_diff(d: dict) -> str:
    bits = []
    if d["gained"]:
        bits.append("gained " + ", ".join(d["gained"]))
    if d["lost"]:
        bits.append("LOST " + ", ".join(d["lost"]))
    return "; ".join(bits) + f"  (n={d['n']})"


@main.command("recall-rls")
@click.option("--queries", default=100, help="Sampled chunk vectors used as queries.")
@click.option("--k", default=50, help="Neighbors per query (the retrieval pool size).")
def recall_rls_cmd(queries: int, k: int):
    """Vector recall under row-level security, per persona, vs exact scan.
    Recorded as an eval run (kind recall_rls) so the dashboard can show it."""
    from raglab import benchmark, eval_retrieval
    from raglab.pipeline import PERSONAS

    receipt = Receipt("raglab recall-rls")
    try:
        with db.connect() as conn:
            tiers = benchmark.recall_under_rls(conn, PERSONAS, n_queries=queries, k=k)
            conn.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())
            run_id = conn.execute(
                "INSERT INTO eval_runs (kind, config_label, git_sha, corpus_hash) "
                "VALUES ('recall_rls', %s, %s, %s) RETURNING id",
                (f"k={k} ef=40", eval_retrieval._git_sha(), eval_retrieval.corpus_hash(conn)),
            ).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(run_id, "all", t.persona, m, v, json.dumps({"visible": t.visible, "total": t.total, "k": t.k}))
                     for t in tiers
                     for m, v in (("recall_mean", t.recall), ("recall_min", t.min_recall),
                                  ("underfilled", t.underfilled), ("median_ms", t.median_ms))],
                )
            conn.commit()
        receipt.add("run id", run_id)
        for t in tiers:
            receipt.add(
                f"{t.persona:9s}",
                f"sees {t.visible}/{t.total} ({t.visible / max(1, t.total):.1%})  "
                f"recall@{t.k} mean {t.recall:.3f} min {t.min_recall:.3f}  "
                f"underfilled {t.underfilled}/{t.n_queries}  median {t.median_ms:.1f} ms",
            )
    except (psycopg.Error, ValueError) as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("identifiers")
def identifiers_cmd():
    """Assign stable member IDs + MRNs to every Synthea patient and rebuild
    the enrollment table (deterministic; safe to re-run)."""
    from raglab import enrollment

    receipt = Receipt("raglab identifiers")
    try:
        with db.connect() as conn:
            stats = enrollment.assign(conn)
            sample = conn.execute(
                "SELECT member_id, mrn FROM synthea.patients ORDER BY id LIMIT 1"
            ).fetchone()
            by_year = conn.execute(
                "SELECT year, count(*) FROM synthea.enrollment GROUP BY year ORDER BY year"
            ).fetchall()
            conn.commit()
        receipt.add("patients", stats["patients"])
        receipt.add("enrollment rows", stats["enrollment_rows"])
        receipt.add("by year", ", ".join(f"{y}: {n}" for y, n in by_year))
        receipt.add("sample", f"member_id {sample[0]}  mrn {sample[1]}")
    except (psycopg.Error, ValueError) as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("eval-diff")
@click.argument("before", type=int)
@click.argument("after", type=int)
def eval_diff_cmd(before: int, after: int):
    """Which golden questions flipped between two retrieval eval runs."""
    from raglab import eval_retrieval

    receipt = Receipt(f"raglab eval-diff {before} {after}")
    try:
        with db.connect() as conn:
            diff = eval_retrieval.diff_runs(conn, before, after)
        if not diff:
            receipt.add("result", "no question flipped on any 0/1 metric")
        for metric, d in diff.items():
            receipt.add(metric, _fmt_diff(d))
    except psycopg.Error as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("bakeoff")
@click.argument("backend_kind", type=click.Choice(["docling", "fast"]))
def bakeoff_cmd(backend_kind: str):
    """Re-ingest the table-heavy 2026 brochures with the chosen parser."""
    from raglab import experiments

    receipt = Receipt(f"raglab bakeoff {backend_kind}")
    try:
        with db.connect() as conn:
            for line in experiments.bakeoff_reingest(conn, backend_kind):
                receipt.add("reingested", line)
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("eval-generation")
@click.option("--label", default="demo", help="config_label recorded with the run.")
def eval_generation_cmd(label: str):
    """Tier-2 end-to-end eval: dual generators, cross-family judged (~$0.05)."""
    from raglab import eval_generation

    receipt = Receipt("raglab eval-generation")
    try:
        with db.connect() as conn:
            result = eval_generation.run(conn, config_label=label)
        receipt.add("run id", result.run_id)
        for generator, metrics in result.by_generator.items():
            receipt.add(generator, "  ".join(f"{m}={v}" for m, v in metrics.items()))
        for row in result.rows:
            flag = "ABSTAINED" if row["abstained"] else f"c={row['correctness']:.2f} f={row['faithfulness']:.2f}"
            receipt.add(f"  {row['question_id']} {row['generator'][:16]}", flag)
        wrong_answers = [
            r for r in result.rows
            if r["category"] == "unanswerable" and not r["abstained"]
        ]
        for row in wrong_answers:
            receipt.fail(f"{row['question_id']}: {row['generator']} answered instead of abstaining")
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("drift")
def drift_cmd():
    """Embedding-drift check vs previous drift run (stable chunk sample)."""
    from raglab import drift

    receipt = Receipt("raglab drift")
    try:
        with db.connect() as conn:
            result = drift.run(conn)
        receipt.add("run id", result.run_id)
        receipt.add("sample", result.sample_size)
        receipt.add("norm mean", f"{result.norm_mean:.4f}")
        receipt.add(
            "centroid shift",
            "first run (no baseline)" if result.centroid_shift is None
            else f"{result.centroid_shift:.5f} (alert > {drift.CENTROID_SHIFT_ALERT})",
        )
        if result.alert:
            receipt.fail(f"embedding drift {result.centroid_shift:.5f} exceeds threshold")
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("dashboard")
def dashboard_cmd():
    """Render the metrics dashboard to data/eval/dashboard.html."""
    from raglab import dashboard

    receipt = Receipt("raglab dashboard")
    try:
        with db.connect() as conn:
            path = dashboard.render(conn)
        receipt.add("dashboard", str(path))
    except Exception as exc:
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


@main.command("snowflake-setup")
def snowflake_setup_cmd():
    """Build Lane 2 end to end: warehouse, roles, tables, masking + row
    access policies, then export Synthea from Postgres and load. Idempotent —
    re-run to rebuild a lapsed trial."""
    from raglab import snowlane

    receipt = Receipt("raglab snowflake-setup")
    try:
        with db.connect() as pg:
            counts = snowlane.export_csvs(pg)
        for name, rows in counts.items():
            receipt.add(f"exported {name}", rows)

        sf = snowlane.connect(bootstrap=True)
        cur = sf.cursor()
        receipt.add("statements run", snowlane.run_setup(cur))
        for name, rows in snowlane.load(cur).items():
            receipt.add(f"loaded {name}", rows)
        snowlane.grant_roles_to_user(cur, os.environ["SNOWFLAKE_USER"])
        receipt.add("roles granted", ", ".join(snowlane.ROLES))
        sf.close()
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("snowflake-verify")
def snowflake_verify_cmd():
    """The entitlement test: same SELECT under every role; per-role result
    shapes must differ exactly as the policies dictate."""
    from raglab import snowlane

    receipt = Receipt("raglab snowflake-verify")
    try:
        sf = snowlane.connect()
        report = snowlane.verify(sf)
        for role, r in report.items():
            name = r["sample_name"] or "-"
            shown = name if len(name) <= 16 else name[:13] + "..."
            receipt.add(
                role,
                f"rows={r['rows']} lobs={r['lobs']} ssn={'Y' if r['ssn_visible'] else 'MASKED'} "
                f"cost={'Y' if r['cost_visible'] else 'MASKED'} "
                f"patients={r['distinct_patients']} name={shown}",
            )

        full = report["CLAIMS_EXAMINER"]
        if not (full["ssn_visible"] and full["cost_visible"]):
            receipt.fail("examiner must see full detail")
        if report["CARE_MANAGER"]["cost_visible"]:
            receipt.fail("care manager saw financial columns")
        if report["ACTUARY"]["ssn_visible"]:
            receipt.fail("actuary saw identifiers")
        with snowlane.connect(role="ACTUARY") as act:
            mid_visible = act.cursor().execute(
                "SELECT count(MEMBER_ID) + count(MRN) FROM PATIENTS"
            ).fetchone()[0]
            enrol = act.cursor().execute("SELECT count(*) FROM ENROLLMENT").fetchone()[0]
        receipt.add("actuary member_id/mrn", f"{mid_visible} visible (must be 0); enrollment rows {enrol}")
        if mid_visible:
            receipt.fail("actuary saw member identifiers")
        if report["ACTUARY"]["distinct_patients"] != full["distinct_patients"]:
            receipt.fail("actuary aggregates must still count distinct members")
        if not (report["PSHB_EXAMINER"]["rows"] < full["rows"]
                and report["PSHB_EXAMINER"]["lobs"] == 1):
            receipt.fail("PSHB examiner must be row-scoped to one book")

        history = snowlane.access_history_peek(sf.cursor())
        receipt.add(
            "access_history",
            f"{len(history)} recent rows" if history
            else "0 rows yet (ACCOUNT_USAGE latency up to ~3h)",
        )
        sf.close()
    except Exception as exc:
        receipt.fail(f"{type(exc).__name__}: {exc}")
    receipt.finish()


@main.command("query")
@click.argument("prompt")
@click.option("--persona", default=None,
              help="public | employee | care_team (omit for admin full view)")
def query_cmd(prompt: str, persona: str | None):
    """Run a prompt through the full funnel and print the context payload —
    exactly what a consuming model receives."""
    import json

    from raglab.pipeline import run_query

    with db.connect() as conn:
        built = run_query(conn, prompt, persona=persona, source="interactive")
    click.echo(json.dumps(built, indent=2, default=str))


@main.command("payload")
@click.argument("payload_id")
def payload_cmd(payload_id: str):
    """Re-display the exact context payload a consumer received, by the
    payload_id recorded in its disclosure. 'Who saw what' — verbatim."""
    import json

    with db.connect() as conn:
        row = conn.execute(
            "SELECT payload, persona, source, asked_at FROM disclosure_log "
            "WHERE payload_id = %s::uuid",
            (payload_id,),
        ).fetchone()
    if row is None:
        raise click.ClickException(f"no disclosure with payload_id {payload_id}")
    payload, persona, source, asked_at = row
    if payload is None:
        raise click.ClickException(
            "disclosure predates payload persistence (metadata only)"
        )
    click.echo(f"# persona={persona} source={source} asked_at={asked_at}")
    click.echo(json.dumps(payload, indent=2, default=str))


@main.command("backup")
@click.option("--tag", default="", help="Suffix for the file name, e.g. pre-pr4.")
def backup_cmd(tag: str):
    """Dump the whole database to data/backups/ (pg_dump inside the container)."""
    from raglab import backup

    receipt = Receipt("raglab backup")
    try:
        path = backup.create(tag)
        receipt.add("file", str(path.relative_to(config.REPO_ROOT)))
        receipt.add("size", f"{path.stat().st_size / 1e6:.1f} MB")
        receipt.add("tables", ", ".join(backup.inventory(path)))
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        receipt.fail(f"{type(exc).__name__}: {detail.decode().strip() or exc}")
    receipt.finish()


@main.command("backups")
def backups_cmd():
    """List backups, newest last."""
    from raglab import backup

    receipt = Receipt("raglab backups")
    files = backup.available()
    for path in files:
        receipt.add(path.name, f"{path.stat().st_size / 1e6:.1f} MB")
    if not files:
        receipt.add("backups", "none yet — run `raglab backup`")
    receipt.finish()


@main.command("restore")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--yes", is_flag=True, help="Required: this replaces the live database.")
def restore_cmd(file: Path, yes: bool):
    """Replace the live database with a backup file."""
    from raglab import backup

    receipt = Receipt(f"raglab restore {file.name}")
    if not yes:
        receipt.fail("refusing without --yes: restore drops and replaces every table")
        receipt.finish()
        return
    try:
        backup.restore(file)
        with db.connect() as conn:
            docs, chunks = conn.execute(
                "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
            ).fetchone()
        receipt.add("restored", f"{docs} documents, {chunks} chunks")
    except (OSError, subprocess.CalledProcessError, psycopg.Error) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        receipt.fail(f"{type(exc).__name__}: {detail.decode().strip() or exc}")
    receipt.finish()
