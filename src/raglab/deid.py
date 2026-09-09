"""PHI de-identification (Presidio) — runs BEFORE indexing, so protected
text never enters the embedding space or the searchable corpus.

Two modes (the posting names both):
- mask:     destructive redaction — entity replaced by its type marker
- tokenize: reversible, CONSISTENT pseudonyms — same original value always
            maps to the same pseudonym (join-able across notes); the
            original is recoverable only via the vault table, which is a
            separate, governable secret.

The claim is a number, not a demo: `raglab deid-eval` scores detection
against the Phase 2.5 injection manifest (recall per entity type + leakage
rate into the indexed corpus) and writes it to the metrics store.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import psycopg

from raglab import config

# Manifest entity types -> Presidio entities we ask the analyzer for.
PRESIDIO_ENTITIES = [
    "PERSON", "DATE_TIME", "US_SSN", "PHONE_NUMBER", "LOCATION",
    "MEDICAL_LICENSE", "US_DRIVER_LICENSE", "ID",
    "MEMBER_ID", "MRN", "CLAIM_ID", "CASE_ID",  # raglab.recognizers
]
# Manifest entity type -> the Presidio entities that count as a CORRECT
# detection. Recall is type-correct: a member ID flagged as a phone number
# is a miss (the vault would file it under the wrong type).
MANIFEST_TO_PRESIDIO = {
    "name": {"PERSON"},
    "date": {"DATE_TIME"},
    "ssn": {"US_SSN"},
    "phone": {"PHONE_NUMBER"},
    "address": {"LOCATION", "PERSON"},  # street lines confuse NER; count either
    "member_id": {"MEMBER_ID"},
    "mrn": {"MRN"},
    "claim_id": {"CLAIM_ID"},
    "case_id": {"CASE_ID"},
}
STRUCTURED = ("member_id", "mrn", "claim_id", "case_id", "ssn", "phone")
# D9 gate, per source: structured identifiers and leakage.
DEID_THRESHOLDS = {"structured_recall": 0.98, "leakage_rate": 0.05}

_analyzer = None


def _get_analyzer(conn=None):
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        from raglab import recognizers

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        engine = provider.create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["en"], nlp_engine=engine)
        # RAGLAB_RECOGNIZERS=off measures stock Presidio alone (the "before").
        if os.environ.get("RAGLAB_RECOGNIZERS", "on") != "off":
            for rec in recognizers.all_recognizers(conn):
                registry.add_recognizer(rec)
        _analyzer = AnalyzerEngine(registry=registry, nlp_engine=engine)
    return _analyzer


def analyze(text: str, conn=None):
    return _get_analyzer(conn).analyze(
        text=text, language="en", entities=PRESIDIO_ENTITIES
    )


MIN_SCORE = 0.35


def resolve_overlaps(results):
    """One replacement per region: keep the highest-scoring result (longer
    on ties) and drop anything overlapping it, plus low-confidence noise."""
    kept = []
    for r in sorted(results, key=lambda r: (-r.score, -(r.end - r.start), r.start)):
        if r.score < MIN_SCORE:
            continue
        if all(r.end <= k.start or r.start >= k.end for k in kept):
            kept.append(r)
    return kept


def canonical_original(entity_type: str, original: str) -> str:
    """Identifiers are vaulted by their canonical value, so every surface
    form of one member ID maps to one pseudonym across all sources."""
    from raglab import identifiers, recognizers

    kind = recognizers.IDENTIFIER_KIND.get(entity_type)
    if kind:
        return identifiers.canonicalize(kind, original) or original
    return original


def _pseudonym(conn: psycopg.Connection, entity_type: str, original: str) -> str:
    """Consistent pseudonym via the vault: same original -> same token."""
    digest = hashlib.sha256(f"{entity_type}|{original}".encode()).hexdigest()
    row = conn.execute(
        "SELECT pseudonym FROM deid_vault WHERE original_hash = %s", (digest,)
    ).fetchone()
    if row:
        return row[0]
    n = conn.execute(
        "SELECT count(*) + 1 FROM deid_vault WHERE entity_type = %s",
        (entity_type,),
    ).fetchone()[0]
    pseudonym = f"[{entity_type}-{n:04d}]"
    conn.execute(
        "INSERT INTO deid_vault (original_hash, entity_type, original, pseudonym) "
        "VALUES (%s, %s, %s, %s)",
        (digest, entity_type, original, pseudonym),
    )
    return pseudonym


def deidentify(conn: psycopg.Connection, text: str, mode: str) -> str:
    """mode: 'mask' | 'tokenize'. Replacements applied right-to-left so
    earlier spans keep their offsets."""
    results = sorted(resolve_overlaps(analyze(text, conn)), key=lambda r: r.start, reverse=True)
    out = text
    for r in results:
        original = canonical_original(r.entity_type, text[r.start:r.end])
        if mode == "mask":
            replacement = f"[{r.entity_type}]"
        else:
            replacement = _pseudonym(conn, r.entity_type, original)
        out = out[: r.start] + replacement + out[r.end:]
    return out


# Query translation covers LOOKUP identifiers only (who/which record).
# DATE_TIME and LOCATION are shared vocabulary — a query saying "2026" means
# the plan year, not some note's tokenized date — so substituting them would
# corrupt ordinary questions.
QUERY_TRANSLATE_TYPES = (
    "PERSON", "US_SSN", "PHONE_NUMBER", "US_DRIVER_LICENSE", "ID",
    "MEDICAL_LICENSE", "MEMBER_ID", "MRN", "CLAIM_ID", "CASE_ID",
)


def translate_query(conn: psycopg.Connection, query: str) -> str:
    """Authorized re-identification bridge: rewrite known PHI originals in a
    query to their vault pseudonyms so tokenized notes stay searchable by the
    identifiers a care team actually uses. The vault is owner-only — callers
    gate this on the persona entitled to re-identification; the persona role
    itself can never read the mapping."""
    rows = conn.execute(
        "SELECT original, pseudonym FROM deid_vault "
        "WHERE entity_type = ANY(%s) "
        "ORDER BY length(original) DESC, original, pseudonym",
        (list(QUERY_TRANSLATE_TYPES),),
    ).fetchall()
    from raglab import identifiers

    out = identifiers.normalize_identifiers(query)  # M048-217-336 -> M048217336
    for original, pseudonym in rows:
        if len(original) < 4:  # short fragments over-match prose
            continue
        out = re.sub(
            rf"\b{re.escape(original)}\b", pseudonym, out, flags=re.IGNORECASE
        )
    return out


@dataclass
class DeidEvalResult:
    # per-source results + gate failures (D9); the top-level fields describe the first source

    run_id: int
    recall_by_type: dict
    overall_recall: float
    leakage_rate: float
    leaked_examples: list
    by_source: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


def _source_text(source, doc: str) -> str | None:
    """The generator-side text of a manifest document: <dir>/<stem>.md, or
    the PDF's text source in <dir>_pdf_src/<stem>.txt."""
    from raglab import config

    stem = doc.rsplit(".", 1)[0]
    base = config.REPO_ROOT / source.dir
    for candidate in (base / f"{stem}.md", base.with_name(base.name + "_pdf_src") / f"{stem}.txt"):
        if candidate.exists():
            return candidate.read_text()
    return None


def evaluate(conn: psycopg.Connection, sample: int | None = None, seed: int = 7,
             gate: bool = False) -> DeidEvalResult:
    """Per source: type-correct detection recall on generator-side text
    (manifest = ground truth) and leakage — manifest entities whose surface
    OR canonical value survives verbatim in an indexed chunk of their own
    document. `sample` = seeded stratified sample size per source (CI)."""
    import random

    from raglab import eval_retrieval, sources
    from raglab.internal_corpus import MANIFEST_DIR

    conn.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())
    registry = sources.load(conn)
    per_source: dict[str, dict] = {}
    for manifest in sorted(MANIFEST_DIR.glob("*.jsonl")):
        source = registry.by_key.get(manifest.stem)
        if source is None:
            continue
        entries = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        if sample and len(entries) > sample:
            entries = random.Random(seed).sample(entries, sample)
        per_type: dict[str, list[int]] = {}
        leaked, total, examples = 0, 0, []
        for entry in entries:
            text = _source_text(source, entry["doc"])
            if text is None:
                continue
            spans = analyze(text, conn)
            for ent in entry["entities"]:
                value, canonical = ent["value"], ent.get("canonical", ent["value"])
                allowed = MANIFEST_TO_PRESIDIO.get(ent["type"], set())
                bucket = per_type.setdefault(ent["type"], [0, 0])
                bucket[1] += 1
                hit, start = False, text.find(value)
                while start != -1 and not hit:
                    end = start + len(value)
                    hit = any(r.start < end and start < r.end and r.entity_type in allowed for r in spans)
                    start = text.find(value, start + 1)
                bucket[0] += int(hit)
                total += 1
                survived = conn.execute(
                    "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
                    "WHERE d.source_path LIKE %s AND (position(%s in c.content) > 0 OR position(%s in c.content) > 0)",
                    (f"%{entry['doc'].rsplit('.', 1)[0]}%", value, canonical),
                ).fetchone()[0]
                if survived:
                    leaked += 1
                    if len(examples) < 10:
                        examples.append((source.key, entry["doc"], ent["type"], value))
        if total:
            per_source[source.key] = {
                "recall_by_type": {t: round(d / n, 3) for t, (d, n) in sorted(per_type.items()) if n},
                "overall_recall": round(sum(d for d, _ in per_type.values()) / sum(n for _, n in per_type.values()), 3),
                "leakage_rate": round(leaked / total, 4),
                "n_entities": total, "n_docs": len(entries), "examples": examples,
            }

    run_id = conn.execute(
        "INSERT INTO eval_runs (kind, config_label, git_sha, corpus_hash) VALUES ('deid', %s, %s, %s) RETURNING id",
        (f"manifest{f'-sample{sample}' if sample else ''}", eval_retrieval._git_sha(), eval_retrieval.corpus_hash(conn)),
    ).fetchone()[0]
    failures = []
    with conn.cursor() as cur:
        for key, r in per_source.items():
            rows = [(run_id, "all", key, f"recall_{t}", v, json.dumps({})) for t, v in r["recall_by_type"].items()]
            rows += [(run_id, "all", key, "overall_recall", r["overall_recall"], json.dumps({"n": r["n_entities"]})),
                     (run_id, "all", key, "leakage_rate", r["leakage_rate"], json.dumps({"examples": r["examples"][:5]}))]
            cur.executemany(
                "INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
            for t, v in r["recall_by_type"].items():
                if t in STRUCTURED and v < DEID_THRESHOLDS["structured_recall"]:
                    failures.append(f"{key}: recall_{t} {v:.3f} < {DEID_THRESHOLDS['structured_recall']}")
            if r["leakage_rate"] >= DEID_THRESHOLDS["leakage_rate"]:
                failures.append(f"{key}: leakage {r['leakage_rate']:.3f} >= {DEID_THRESHOLDS['leakage_rate']}")
    conn.commit()
    all_types: dict[str, list[int]] = {}
    overall = 0.0
    if per_source:
        overall = round(sum(r["overall_recall"] * r["n_entities"] for r in per_source.values())
                        / sum(r["n_entities"] for r in per_source.values()), 3)
    first = next(iter(per_source.values()), {"recall_by_type": {}, "leakage_rate": 0.0, "examples": []})
    return DeidEvalResult(
        run_id=run_id, recall_by_type=first["recall_by_type"], overall_recall=overall,
        leakage_rate=first["leakage_rate"], leaked_examples=first["examples"],
        by_source=per_source, failures=failures if gate else [],
    )
