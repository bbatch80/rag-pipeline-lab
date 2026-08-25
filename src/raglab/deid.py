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
import re
from dataclasses import dataclass

import psycopg

from raglab import config
from raglab.synth.notes import MANIFEST_PATH

# Manifest entity types -> Presidio entities we ask the analyzer for.
PRESIDIO_ENTITIES = [
    "PERSON", "DATE_TIME", "US_SSN", "PHONE_NUMBER", "LOCATION",
    "MEDICAL_LICENSE", "US_DRIVER_LICENSE", "ID",
]
MANIFEST_TO_PRESIDIO = {
    "name": {"PERSON"},
    "date": {"DATE_TIME"},
    "ssn": {"US_SSN"},
    "phone": {"PHONE_NUMBER"},
    "address": {"LOCATION", "PERSON"},  # street lines confuse NER; count either
    "member_id": {"ID", "US_DRIVER_LICENSE", "US_SSN", "PHONE_NUMBER"},
    "mrn": {"ID", "US_DRIVER_LICENSE"},
}

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    return _analyzer


def analyze(text: str):
    return _get_analyzer().analyze(
        text=text, language="en", entities=PRESIDIO_ENTITIES
    )


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
    results = sorted(analyze(text), key=lambda r: r.start, reverse=True)
    out = text
    for r in results:
        original = text[r.start:r.end]
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
    "MEDICAL_LICENSE",
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
    out = query
    for original, pseudonym in rows:
        if len(original) < 4:  # short fragments over-match prose
            continue
        out = re.sub(
            rf"\b{re.escape(original)}\b", pseudonym, out, flags=re.IGNORECASE
        )
    return out


@dataclass
class DeidEvalResult:
    run_id: int
    recall_by_type: dict
    overall_recall: float
    leakage_rate: float
    leaked_examples: list


def evaluate(conn: psycopg.Connection) -> DeidEvalResult:
    """Detection recall on note SOURCE text (manifest = ground truth), plus
    leakage rate: manifest entities surviving verbatim in the INDEXED corpus."""
    from raglab import eval_retrieval
    from raglab.synth.notes import NOTES_DIR, PDF_SRC_DIR

    conn.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())
    entries = [json.loads(l) for l in MANIFEST_PATH.read_text().splitlines() if l.strip()]

    per_type = {}  # type -> [detected, total]
    for entry in entries:
        stem = entry["doc"].rsplit(".", 1)[0]
        src = NOTES_DIR / f"{stem}.md"
        if not src.exists():
            src = PDF_SRC_DIR / f"{stem}.txt"
        text = src.read_text()
        spans = analyze(text)
        detected_ranges = [(r.start, r.end) for r in spans]
        for ent in entry["entities"]:
            value = ent["value"]
            bucket = per_type.setdefault(ent["type"], [0, 0])
            bucket[1] += 1
            start = text.find(value)
            hit = False
            while start != -1 and not hit:
                end = start + len(value)
                hit = any(s < end and start < e for s, e in detected_ranges)
                start = text.find(value, start + 1)
            bucket[0] += int(hit)

    leaked, total, examples = 0, 0, []
    for entry in entries:
        for ent in entry["entities"]:
            total += 1
            row = conn.execute(
                "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.source_path LIKE %s AND position(%s in c.content) > 0",
                (f"%{entry['doc'].rsplit('.', 1)[0]}%", ent["value"]),
            ).fetchone()[0]
            if row:
                leaked += 1
                if len(examples) < 10:
                    examples.append((entry["doc"], ent["type"], ent["value"]))

    recall_by_type = {
        t: round(d / n, 3) for t, (d, n) in sorted(per_type.items()) if n
    }
    overall = round(
        sum(d for d, _ in per_type.values()) / sum(n for _, n in per_type.values()), 3
    )
    leakage = round(leaked / total, 4) if total else 0.0

    run_id = conn.execute(
        "INSERT INTO eval_runs (kind, config_label, git_sha) "
        "VALUES ('deid', 'manifest', %s) RETURNING id",
        (eval_retrieval._git_sha(),),
    ).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO eval_scores (run_id, question_id, category, metric, value, detail) "
            "VALUES (%s, '-', 'deid', %s, %s, %s)",
            [(run_id, f"recall_{t}", v, "{}") for t, v in recall_by_type.items()]
            + [
                (run_id, "overall_recall", overall, "{}"),
                (run_id, "leakage_rate", leakage,
                 json.dumps({"examples": examples[:5]})),
            ],
        )
    conn.commit()
    return DeidEvalResult(
        run_id=run_id, recall_by_type=recall_by_type,
        overall_recall=overall, leakage_rate=leakage, leaked_examples=examples,
    )
