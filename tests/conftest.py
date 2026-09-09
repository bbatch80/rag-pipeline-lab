import os
os.environ.setdefault("RAGLAB_EMBED_CACHE", "off")  # tests patch the embedder; the cache must not answer for it
import pytest

from raglab import db as raglab_db


def _refuse_commit():
    raise RuntimeError("tests run inside one rolled-back transaction and must never commit")


@pytest.fixture
def db():
    """Connection whose work is always rolled back — tests leave no residue.
    Tables are emptied inside the transaction so tests see a clean slate
    regardless of what the live database holds; rollback restores it.
    commit() is disabled while the test runs: a single commit would make the
    emptying permanent and wipe the live corpus (it did once, 2026-09-08)."""
    with raglab_db.connect() as conn:
        real_commit = conn.commit
        conn.commit = _refuse_commit
        try:
            conn.execute("DELETE FROM quarantine")
            conn.execute("DELETE FROM documents")
            yield conn
        finally:
            conn.rollback()
            conn.commit = real_commit  # the context manager commits on exit; nothing is pending
