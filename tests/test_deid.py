"""Custom recognizers, canonical vaulting, query normalization, and the
type-correct per-source de-id eval. Skipped where Presidio is not installed."""

import json

import pytest

pytest.importorskip("presidio_analyzer")

from raglab import deid, identifiers  # noqa: E402


class _NoCommit:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, *a, **k):
        return self._conn.execute(*a, **k)

    def cursor(self, *a, **k):
        return self._conn.cursor(*a, **k)

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.mark.parametrize("style", ["canonical", "grouped", "spaced"])
def test_member_id_detected_in_every_style(db, style):
    mid = identifiers.member_id("k")
    text = f"member {identifiers.present('member_id', mid, style)} called about a claim"
    hits = [r for r in deid.analyze(text, db) if r.entity_type == "MEMBER_ID"]
    assert hits and deid.canonical_original("MEMBER_ID", text[hits[0].start:hits[0].end]) == mid


def test_bare_member_id_needs_context_and_check_digit(db):
    mid = identifiers.member_id("k")
    with_ctx = f"member id {mid[1:]} on file"
    assert any(r.entity_type == "MEMBER_ID" for r in deid.analyze(with_ctx, db))
    bad = mid[:-1] + str((int(mid[-1]) + 1) % 10)
    assert not any(r.entity_type == "MEMBER_ID" for r in deid.analyze(f"member id {bad}", db))


def test_mrn_claim_case_and_shorthand_dates(db):
    mrn, clm, apl = identifiers.mrn("k"), identifiers.claim_id("e"), identifiers.case_id(3)
    text = f"{identifiers.present('mrn', mrn, 'bare')} seen 02.11.24 and Feb 11 2024; claim {clm}; appeal {apl}"
    types = {r.entity_type for r in deid.analyze(text, db)}
    assert {"MRN", "CLAIM_ID", "CASE_ID", "DATE_TIME"} <= types


def test_variants_share_one_pseudonym(db):
    mid = identifiers.member_id("k")
    a = deid.deidentify(_NoCommit(db), f"member {identifiers.present('member_id', mid, 'grouped')}", "tokenize")
    b = deid.deidentify(_NoCommit(db), f"member {identifiers.present('member_id', mid, 'spaced')}", "tokenize")
    assert "[MEMBER_ID-" in a and a == b


def test_overlaps_resolve_to_one_replacement(db):
    clm = identifiers.claim_id("e")
    out = deid.deidentify(_NoCommit(db), f"claim {clm} denied", "mask")
    assert out.count("[") == 1 and "[CLAIM_ID]" in out


def test_query_translation_normalizes_surface_forms(db):
    mid = identifiers.member_id("k")
    deid.deidentify(_NoCommit(db), f"member {mid}", "tokenize")  # vault the canonical value
    translated = deid.translate_query(_NoCommit(db), f"notes for M{mid[1:4]}-{mid[4:7]}-{mid[7:]}")
    assert "[MEMBER_ID-" in translated


def test_evaluate_is_type_correct_and_per_source(db, tmp_path, monkeypatch):
    from raglab import internal_corpus
    mid, mrn = identifiers.member_id("p1"), identifiers.mrn("p1")
    text = f"pt member {identifiers.present('member_id', mid, 'grouped')} {mrn} seen 3/4/24"
    (tmp_path / "clinical_notes.jsonl").write_text(json.dumps({
        "doc": "note_0001_soap.md", "patient_id": "p1", "template": "soap",
        "entities": [{"type": "member_id", "value": identifiers.present("member_id", mid, "grouped"), "canonical": mid},
                     {"type": "mrn", "value": mrn, "canonical": mrn},
                     {"type": "date", "value": "3/4/24", "canonical": "3/4/24"}],
    }) + "\n")
    monkeypatch.setattr(internal_corpus, "MANIFEST_DIR", tmp_path)
    monkeypatch.setattr(deid, "_source_text", lambda source, doc: text)
    # one indexed chunk still carrying the canonical member ID = a leak
    doc = db.execute("INSERT INTO documents (source_path, title, content_hash, source_id) "
                     "VALUES ('data/internal/notes/note_0001_soap.md', 'n', 'h', 7) RETURNING id").fetchone()[0]
    db.execute("INSERT INTO chunks (document_id, chunk_index, content, doc_type) VALUES (%s, 0, %s, 'clinical_note')",
               (doc, f"tokenized text but {mid} survived"))
    result = deid.evaluate(_NoCommit(db), gate=True)
    r = result.by_source["clinical_notes"]
    assert r["recall_by_type"]["member_id"] == 1.0 and r["recall_by_type"]["mrn"] == 1.0
    assert r["recall_by_type"]["date"] == 1.0
    assert r["leakage_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert any("leakage" in f for f in result.failures)


def test_non_phi_spans_are_released():
    """Over-redaction control: domain vocabulary flagged as a name or place,
    and date-like spans that are not specific dates, protect nothing and are
    released; real names, places, and dated elements stay."""
    from raglab.deid import is_phi_span

    released = [("PERSON", "advd"), ("PERSON", "APPEAL_INFO mbr"), ("LOCATION", "PA"),
                ("PERSON", "card req"), ("PERSON", "Jan"), ("DATE_TIME", "2025"),
                ("DATE_TIME", "30 day"), ("DATE_TIME", "last week"), ("DATE_TIME", "Outpatient")]
    kept = [("PERSON", "Tristan Tillman"), ("PERSON", "Mueller"), ("LOCATION", "Kansas City"),
            ("DATE_TIME", "Aug 31 2026"), ("DATE_TIME", "Sep 19 1985"), ("DATE_TIME", "08/31/26"),
            ("DATE_TIME", "Jan 2026"), ("DATE_TIME", "2026-08-31"), ("MEMBER_ID", "M822099594")]
    assert not any(is_phi_span(t, s) for t, s in released), [s for t, s in released if is_phi_span(t, s)]
    assert all(is_phi_span(t, s) for t, s in kept), [s for t, s in kept if not is_phi_span(t, s)]


def test_resolve_overlaps_releases_non_phi_when_given_text():
    from presidio_analyzer import RecognizerResult

    from raglab.deid import resolve_overlaps

    text = "mbr advd PA required; DOB Sep 19 1985; window 30 day"
    results = [
        RecognizerResult("PERSON", text.index("advd"), text.index("advd") + 4, 0.85),
        RecognizerResult("LOCATION", text.index("PA"), text.index("PA") + 2, 0.85),
        RecognizerResult("DATE_TIME", text.index("Sep"), text.index("1985") + 4, 0.6),
        RecognizerResult("DATE_TIME", text.index("30 day"), text.index("30 day") + 6, 0.6),
    ]
    assert {r.entity_type for r in resolve_overlaps(results)} == {"PERSON", "LOCATION", "DATE_TIME"}
    kept = resolve_overlaps(results, text)
    assert [text[r.start:r.end] for r in kept] == ["Sep 19 1985"]
