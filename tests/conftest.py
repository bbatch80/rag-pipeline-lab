import os
os.environ.setdefault("RAGLAB_EMBED_CACHE", "off")  # tests patch the embedder; the cache must not answer for it
os.environ.setdefault("RAGLAB_RERANK_CACHE", "off")  # tests patch the reranker; same reason
os.environ.setdefault("RAGLAB_PLANNER", "rules")  # tests never call the planner model; model tests inject a fake client
os.environ.setdefault("RAGLAB_PLAN_CACHE", "off")
import pytest

from raglab import db as raglab_db


def _refuse_commit():
    raise RuntimeError("tests run inside one rolled-back transaction and must never commit")


@pytest.fixture
def db(request):
    """Connection whose work is always rolled back — tests leave no residue.
    By default the document tables are emptied inside the transaction so a
    test sees a clean slate regardless of what the live database holds;
    rollback restores it. That DELETE cascades through ~25k chunks and costs
    ~2 s per test, so modules that only READ the corpus (or only touch the
    synthea tables) declare `pytestmark = pytest.mark.readonly` and skip it;
    a single test in such a module can ask for the clean slate back with
    `@pytest.mark.clean_corpus`.
    commit() is disabled while the test runs: a single commit would make the
    emptying permanent and wipe the live corpus (it did once, 2026-09-08)."""
    clean = bool(request.node.get_closest_marker("clean_corpus")) or not request.node.get_closest_marker("readonly")
    with raglab_db.connect() as conn:
        real_commit = conn.commit
        conn.commit = _refuse_commit
        try:
            if clean:
                conn.execute("DELETE FROM quarantine")
                conn.execute("DELETE FROM documents")
            yield conn
        finally:
            conn.rollback()
            conn.commit = real_commit  # the context manager commits on exit; nothing is pending
