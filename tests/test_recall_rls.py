"""Vector recall under RLS is measured per persona against an exact scan run
as the same persona, on a synthetic corpus where care_team sees ~3% of rows."""

import random

from raglab import benchmark


def test_recall_under_rls_per_persona(db):
    rng = random.Random(11)
    big = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
        "VALUES ('t/big.md', 'big', 'h', 'employee', 3) RETURNING id"
    ).fetchone()[0]
    notes = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, acl_tag, source_id) "
        "VALUES ('t/notes.md', 'notes', 'h', 'care_team', 7) RETURNING id"
    ).fetchone()[0]

    def vec():
        base = [rng.uniform(-1, 1) for _ in range(4)]
        return "[" + ",".join(f"{base[i % 4] + rng.gauss(0, 0.05):.5f}" for i in range(1536)) + "]"

    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type, embedding) "
            "VALUES (%s, %s, 'x', 'employee', 2026, 'sop', %s::vector)",
            [(big, i, vec()) for i in range(300)],
        )
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, acl_tag, year, doc_type, embedding) "
            "VALUES (%s, %s, 'note', 'care_team', 2026, 'clinical_note', %s::vector)",
            [(notes, i, vec()) for i in range(10)],
        )
    db.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    db.execute("SET LOCAL max_parallel_maintenance_workers = 0")
    db.execute("CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)")

    tiers = {t.persona: t for t in benchmark.recall_under_rls(db, ("public", "employee", "care_team"), n_queries=20, k=5)}
    assert tiers["admin"].visible == 310 and tiers["employee"].visible == 300
    assert tiers["public"].visible == 0 and tiers["care_team"].visible == 10
    assert tiers["employee"].recall >= 0.9
    assert tiers["care_team"].recall >= 0.9 and tiers["care_team"].underfilled == 0, "3% visibility must still fill k"
