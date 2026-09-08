"""Schema smoke tests: vector ordering, full-text ranking, cascade deletes."""

import random

from raglab import config


def _random_vector(seed: int) -> str:
    rng = random.Random(seed)
    values = [rng.uniform(-1, 1) for _ in range(config.EMBEDDING_DIMENSIONS)]
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def _insert_document(db, source_path: str) -> int:
    return db.execute(
        """
        INSERT INTO documents (source_path, title, content_hash, source_id)
        VALUES (%s, %s, %s, 1) RETURNING id
        """,
        (source_path, "Smoke Test Doc", "hash-smoke"),
    ).fetchone()[0]


def test_vector_ordering(db):
    doc_id = _insert_document(db, "smoke/vector.pdf")
    for i in range(5):
        db.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, doc_type, embedding)
            VALUES (%s, %s, %s, 'brochure', %s::vector)
            """,
            (doc_id, i, f"chunk {i}", _random_vector(seed=i)),
        )

    query_vector = _random_vector(seed=3)
    rows = db.execute(
        """
        SELECT chunk_index, embedding <=> %s::vector AS distance
        FROM chunks WHERE document_id = %s
        ORDER BY distance LIMIT 5
        """,
        (query_vector, doc_id),
    ).fetchall()

    assert rows[0][0] == 3, "identical vector must rank first"
    assert rows[0][1] < 1e-6, "cosine distance to itself must be ~0"
    distances = [r[1] for r in rows]
    assert distances == sorted(distances)


def test_fulltext_ranking(db):
    doc_id = _insert_document(db, "smoke/fulltext.pdf")
    contents = [
        "The deductible applies before the plan pays. Deductible amounts vary.",
        "Preventive care is covered in full with no member cost share.",
        "A deductible is listed in the benefits summary.",
    ]
    for i, content in enumerate(contents):
        db.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, doc_type)
            VALUES (%s, %s, %s, 'brochure')
            """,
            (doc_id, i, content),
        )

    rows = db.execute(
        """
        SELECT chunk_index,
               -(content <@> to_bm25query('deductible', 'chunks_bm25_idx')) AS rank
        FROM chunks
        WHERE document_id = %s
          AND -(content <@> to_bm25query('deductible', 'chunks_bm25_idx')) > 0
        ORDER BY rank DESC
        """,
        (doc_id,),
    ).fetchall()

    assert len(rows) == 2, "only chunks mentioning the term should match"
    assert rows[0][0] == 0, "chunk with two mentions must outrank one mention"


def test_cascade_delete(db):
    doc_id = _insert_document(db, "smoke/cascade.pdf")
    db.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, doc_type) VALUES (%s, 0, 'x', 'brochure')",
        (doc_id,),
    )
    db.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    remaining = db.execute(
        "SELECT count(*) FROM chunks WHERE document_id = %s", (doc_id,)
    ).fetchone()[0]
    assert remaining == 0
