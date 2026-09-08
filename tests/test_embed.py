"""Embed contract tests: the embed-where-NULL resume property, verified
against a stub client — CI never calls the paid API."""

from types import SimpleNamespace

from raglab.embed import embed_pending


class _NoCommit:
    """Suppresses commit so embed_pending's per-batch commits don't break
    the rollback-based test fixture."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def commit(self):
        pass


class FakeClient:
    def __init__(self):
        self.batches: list[list[str]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.batches.append(list(input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.001 * (i + 1)] * 1536)
                for i in range(len(input))
            ],
            usage=SimpleNamespace(total_tokens=len(input) * 10),
        )


def _seed_chunks(db, n_null: int, n_embedded: int) -> int:
    doc_id = db.execute(
        "INSERT INTO documents (source_path, title, content_hash, source_id) "
        "VALUES ('t/e.pdf', 'T', 'h', 1) RETURNING id"
    ).fetchone()[0]
    vec = "[" + ",".join(["0.5"] * 1536) + "]"
    for i in range(n_null + n_embedded):
        db.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, doc_type, embedding) "
            "VALUES (%s, %s, %s, 'brochure', %s)",
            (doc_id, i, f"chunk {i}", vec if i < n_embedded else None),
        )
    return doc_id


def test_embed_selects_only_null_and_resumes(db):
    _seed_chunks(db, n_null=5, n_embedded=2)
    client = FakeClient()
    conn = _NoCommit(db)

    stats = embed_pending(conn, client, batch_size=2)
    assert stats.embedded == 5
    assert stats.batches == 3  # 2 + 2 + 1
    sent = [text for batch in client.batches for text in batch]
    assert len(sent) == 5, "pre-embedded chunks must never be re-sent"

    remaining = db.execute(
        "SELECT count(*) FROM chunks WHERE embedding IS NULL"
    ).fetchone()[0]
    assert remaining == 0

    rerun = embed_pending(conn, client, batch_size=2)
    assert rerun.embedded == 0 and rerun.batches == 0, "re-run must be a no-op"
