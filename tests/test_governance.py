"""Governance contract tests — the phase's exit criteria as hard assertions.

The db fixture empties tables in its rolled-back transaction; these tests
insert their own tiered fixtures. RLS policies and personas are cluster/
table-level and already applied.
"""

import pytest

# Every document is bound to a source row (migration 002); the tier decides which.
SOURCE_FOR_TAG = {"public": 1, "employee": 3, "care_team": 7, "member_services": 8, "appeals": 9}
DOC_TYPE_FOR_TAG = {"public": "brochure", "employee": "sop", "care_team": "clinical_note",
                    "member_services": "call_note", "appeals": "appeal"}

from raglab.pipeline import run_query


def _seed_tiers(db, per_tier=3, embed=False):
    doc_ids = {}
    vec = "[" + ",".join(["0.5"] * 1536) + "]"
    for tag in ("public", "employee", "care_team", "member_services", "appeals"):
        doc_id = db.execute(
            "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
            "VALUES (%s, %s, 'h', %s, %s) RETURNING id",
            (f"t/{tag}.md", f"doc-{tag}", tag, SOURCE_FOR_TAG[tag]),
        ).fetchone()[0]
        doc_ids[tag] = doc_id
        for i in range(per_tier):
            db.execute(
                "INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type, embedding) "
                "VALUES (%s, %s, %s, %s, 2026, %s, %s)",
                (doc_id, i, f"{tag} secret fact {i}", tag, DOC_TYPE_FOR_TAG[tag],
                 vec if embed else None),
            )
    return doc_ids


VISIBILITY = {
    "persona_public": {"public"},
    "persona_employee": {"public", "employee"},
    "persona_care_team": {"public", "care_team"},
    # Phase 2 tiers inherit persona_employee (decision 1)
    "persona_member_services": {"public", "employee", "member_services"},
    "persona_appeals": {"public", "employee", "appeals"},
}


@pytest.mark.parametrize("role,expected_tags", VISIBILITY.items())
def test_lateral_entitlement_matrix(db, role, expected_tags):
    _seed_tiers(db)
    db.execute(f"SET LOCAL ROLE {role}")
    visible = {
        r[0] for r in db.execute("SELECT DISTINCT acl_tag FROM chunks").fetchall()
    }
    db.execute("RESET ROLE")
    assert visible == expected_tags


def test_care_team_chunk_invisible_to_employee(db):
    _seed_tiers(db)
    db.execute("SET LOCAL ROLE persona_employee")
    leaked = db.execute(
        "SELECT count(*) FROM chunks WHERE content LIKE 'care_team secret%'"
    ).fetchone()[0]
    db.execute("RESET ROLE")
    assert leaked == 0, "clinical content must not exist for the employee role"


def test_employee_chunk_invisible_to_care_team(db):
    _seed_tiers(db)
    db.execute("SET LOCAL ROLE persona_care_team")
    leaked = db.execute(
        "SELECT count(*) FROM chunks WHERE content LIKE 'employee secret%'"
    ).fetchone()[0]
    db.execute("RESET ROLE")
    assert leaked == 0


def test_revoke_changes_visibility_without_reindex(db):
    """The security-trimming demo as a test: a non-superuser agent loses a
    group membership; the same query returns fewer rows instantly — no
    index rebuild, no corpus change. (CREATE ROLE is transactional; the
    fixture rollback removes the demo agent.)"""
    _seed_tiers(db)
    db.execute("CREATE ROLE demo_agent NOLOGIN")
    db.execute("GRANT persona_public, persona_employee TO demo_agent")

    db.execute("SET LOCAL ROLE demo_agent")
    before = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    db.execute("RESET ROLE")
    assert before == 6  # 3 public + 3 employee

    db.execute("REVOKE persona_employee FROM demo_agent")

    db.execute("SET LOCAL ROLE demo_agent")
    after = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    tags = {r[0] for r in db.execute("SELECT DISTINCT acl_tag FROM chunks").fetchall()}
    db.execute("RESET ROLE")
    assert after == 3 and tags == {"public"}, (
        "revoked membership must trim rows at query time"
    )


@pytest.mark.slow
def test_hnsw_survives_heavy_rls_trimming(db):
    """Starvation check: a persona seeing ~3% of rows must still get k
    results from an index scan (iterative scan walks past invisible rows)."""
    import random

    rng = random.Random(7)
    doc_id = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
        "VALUES ('t/big.md', 'big', 'h', 'public', 1) RETURNING id"
    ).fetchone()[0]
    note_doc = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
        "VALUES ('t/notes.md', 'notes', 'h', 'care_team', 7) RETURNING id"
    ).fetchone()[0]

    def vec():
        base = [rng.uniform(-1, 1) for _ in range(4)]
        return "[" + ",".join(
            f"{base[i % 4] + rng.gauss(0, 0.05):.5f}" for i in range(1536)
        ) + "]"

    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type, embedding) "
            "VALUES (%s, %s, 'x', 'employee', 2026, 'sop', %s::vector)",
            [(doc_id, i, vec()) for i in range(300)],
        )
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type, embedding) "
            "VALUES (%s, %s, 'note', 'care_team', 2026, 'clinical_note', %s::vector)",
            [(note_doc, i, vec()) for i in range(10)],
        )
    db.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    db.execute("SET LOCAL max_parallel_maintenance_workers = 0")
    db.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    query = vec()
    db.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
    db.execute("SET LOCAL enable_seqscan = off")
    db.execute("SET LOCAL ROLE persona_care_team")
    rows = db.execute(
        "SELECT id FROM chunks WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector LIMIT 8",
        (query,),
    ).fetchall()
    db.execute("RESET ROLE")
    assert len(rows) == 8, (
        f"index scan starved under RLS: got {len(rows)}/8 "
        "(care_team sees only 10 of 310 rows)"
    )


def test_vault_translation_follows_entitlement(db, monkeypatch):
    """Tokenized notes are searchable by real identifiers ONLY for sessions
    entitled to the vault (admin as owner, care_team by grant): their queries
    are rewritten name -> pseudonym; other personas search the literal
    (absent) name."""
    from raglab import deid

    db.execute(
        "INSERT INTO deid_vault (original_hash, entity_type, original, pseudonym) "
        "VALUES ('h1', 'PERSON', 'Quorthon', '[PERSON-9999]')"
    )
    assert deid.translate_query(db, "asthma patient Quorthon?") == (
        "asthma patient [PERSON-9999]?"
    )
    assert deid.translate_query(db, "the Quorthonian era") == (
        "the Quorthonian era"
    ), "substitution must respect word boundaries"

    _seed_tiers(db, embed=True)
    seen = {}

    def fake_embed(text):
        seen["query"] = text
        return "[" + ",".join(["0.5"] * 1536) + "]"

    monkeypatch.setattr("raglab.retrieval.embed_query", fake_embed)

    class _FakeModel:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    import raglab.rerank as rr

    monkeypatch.setattr(rr, "_model", _FakeModel())

    class _NoCommit:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *a, **k):
            return self._conn.execute(*a, **k)

        def commit(self):
            pass

    run_query(_NoCommit(db), "notes on Quorthon", persona="care_team")
    assert "[PERSON-9999]" in seen["query"]
    run_query(_NoCommit(db), "notes on Quorthon")  # admin owns the vault
    assert "[PERSON-9999]" in seen["query"]
    run_query(_NoCommit(db), "notes on Quorthon", persona="employee")
    assert "Quorthon" in seen["query"] and "[PERSON-9999]" not in seen["query"]


def test_disclosure_record_survives_document_deletion(db, monkeypatch):
    _seed_tiers(db, embed=True)
    monkeypatch.setattr(
        "raglab.retrieval.embed_query",
        lambda text: "[" + ",".join(["0.5"] * 1536) + "]",
    )

    class _FakeModel:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    import raglab.rerank as rr

    monkeypatch.setattr(rr, "_model", _FakeModel())

    class _NoCommit:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *a, **k):
            return self._conn.execute(*a, **k)

        def commit(self):
            pass

    built = run_query(_NoCommit(db), "what is the secret fact?",
                      persona="employee", source="eval")
    assert built["status"] == "ok"
    record = db.execute(
        "SELECT chunk_ids, doc_titles, acl_basis, payload FROM disclosure_log "
        "WHERE payload_id = %s", (built["payload_id"],)
    ).fetchone()
    assert record is not None and record[0], "disclosure must be recorded"
    assert set(record[2]) <= {"public", "employee"}, (
        "employee persona must never disclose care_team content"
    )
    assert record[3] and record[3]["payload_id"] == built["payload_id"], (
        "the exact delivered payload must be reproducible from the disclosure"
    )

    db.execute("DELETE FROM documents")
    survivor = db.execute(
        "SELECT doc_titles FROM disclosure_log WHERE payload_id = %s",
        (built["payload_id"],),
    ).fetchone()
    assert survivor is not None and survivor[0], (
        "audit record must survive corpus deletion"
    )


@pytest.mark.parametrize("role,hidden", [
    ("persona_employee", {"member_services", "appeals", "care_team"}),
    ("persona_member_services", {"appeals", "care_team"}),
    ("persona_appeals", {"member_services", "care_team"}),
    ("persona_care_team", {"member_services", "appeals", "employee"}),
])
def test_phase2_walls_hold_both_ways(db, role, hidden):
    """The operations tiers inherit employee and see their own records only;
    the clinical tier sees neither; employee sees neither new tier."""
    _seed_tiers(db)
    db.execute(f"SET LOCAL ROLE {role}")
    for tag in hidden:
        leaked = db.execute("SELECT count(*) FROM chunks WHERE content LIKE %s", (f"{tag} secret%",)).fetchone()[0]
        assert leaked == 0, f"{role} must not see {tag} content"
    db.execute("RESET ROLE")


def test_new_tiers_inherit_employee_material(db):
    _seed_tiers(db)
    for role in ("persona_member_services", "persona_appeals"):
        db.execute(f"SET LOCAL ROLE {role}")
        assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'employee secret%'").fetchone()[0] == 3, role
        db.execute("RESET ROLE")


def _link(db, title, case="APL-TEST001", kind="clinical_note"):
    db.execute("INSERT INTO appeal_evidence (case_id, document_title, kind) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", (case, title, kind))


def test_relational_entitlement_follows_the_evidence_link(db):
    """A clinical note is visible to the appeals tier while an appeal cites it
    — and only that note; unlinking it removes it with zero re-indexing."""
    _seed_tiers(db)  # care_team doc 'doc-care_team' with 3 chunks, unlinked
    other = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
        "VALUES ('t/other.md', 'note_other', 'h', 'care_team', 7) RETURNING id").fetchone()[0]
    db.execute("INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type) "
               "VALUES (%s, 0, 'care_team other note', 'care_team', 2026, 'clinical_note')", (other,))
    _link(db, "doc-care_team")
    db.execute("SET LOCAL ROLE persona_appeals")
    assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'care_team secret%'").fetchone()[0] == 3, "linked note visible"
    assert db.execute("SELECT count(*) FROM chunks WHERE content = 'care_team other note'").fetchone()[0] == 0, "unlinked note invisible"
    assert db.execute("SELECT count(*) FROM documents WHERE title = 'doc-care_team'").fetchone()[0] == 1
    db.execute("RESET ROLE")
    for role in ("persona_member_services", "persona_employee", "persona_public"):
        db.execute(f"SET LOCAL ROLE {role}")
        assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'care_team secret%'").fetchone()[0] == 0, f"{role} never sees clinical notes"
        db.execute("RESET ROLE")
    db.execute("DELETE FROM appeal_evidence WHERE case_id = 'APL-TEST001'")
    db.execute("SET LOCAL ROLE persona_appeals")
    assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'care_team secret%'").fetchone()[0] == 0, "unlinked -> gone, no re-index"
    db.execute("RESET ROLE")


def test_relational_entitlement_covers_cited_call_notes(db):
    """Decision 6: a call note is visible to the appeals tier while an appeal
    cites it (kind='call_note'); uncited calls stay invisible; the kind must
    match — a call note cited as 'clinical_note' opens nothing; no other tier
    gains anything from the link; unlinking removes it with zero re-indexing."""
    _seed_tiers(db)  # member_services doc 'doc-member_services' with 3 chunks, unlinked
    other = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
        "VALUES ('t/other-call.md', 'call_other', 'h', 'member_services', %s) RETURNING id",
        (SOURCE_FOR_TAG["member_services"],)).fetchone()[0]
    db.execute("INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type) "
               "VALUES (%s, 0, 'member_services other call', 'member_services', 2026, 'call_note')", (other,))
    _link(db, "call_other", kind="clinical_note")  # wrong kind: must not open the call
    _link(db, "doc-member_services", kind="call_note")
    db.execute("SET LOCAL ROLE persona_appeals")
    assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'member_services secret%'").fetchone()[0] == 3, "cited call visible"
    assert db.execute("SELECT count(*) FROM chunks WHERE content = 'member_services other call'").fetchone()[0] == 0, "kind mismatch opens nothing"
    assert db.execute("SELECT count(*) FROM documents WHERE title = 'doc-member_services'").fetchone()[0] == 1
    db.execute("RESET ROLE")
    for role in ("persona_care_team", "persona_employee", "persona_public"):
        db.execute(f"SET LOCAL ROLE {role}")
        assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'member_services secret%'").fetchone()[0] == 0, f"{role} never sees call notes"
        db.execute("RESET ROLE")
    db.execute("DELETE FROM appeal_evidence WHERE case_id = 'APL-TEST001'")
    db.execute("SET LOCAL ROLE persona_appeals")
    assert db.execute("SELECT count(*) FROM chunks WHERE content LIKE 'member_services secret%'").fetchone()[0] == 0, "unlinked -> gone, no re-index"
    db.execute("RESET ROLE")


def test_person_record_sources_are_member_scoped(db):
    """Every source whose documents are records about one person carries the
    member_scoped flag, so the member filter applies (golden S1 found clinical
    notes registered without it: a chart question returned other patients)."""
    rows = db.execute(
        "SELECT key, member_scoped FROM sources WHERE key IN ('call_notes', 'appeal_documents', 'clinical_notes')"
    ).fetchall()
    assert len(rows) == 3
    assert all(flag for _, flag in rows), [k for k, flag in rows if not flag]
