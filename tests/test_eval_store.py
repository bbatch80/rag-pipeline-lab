"""The eval store names the corpus it measured."""

from raglab import eval_retrieval


def test_corpus_hash_is_stable_and_tracks_documents(db):
    empty = eval_retrieval.corpus_hash(db)
    assert empty == eval_retrieval.corpus_hash(db), "same corpus, same digest"

    db.execute(
        "INSERT INTO documents (source_path, title, content_hash) "
        "VALUES ('x/a.md', 'A', 'hash-a')"
    )
    one = eval_retrieval.corpus_hash(db)
    assert one != empty, "adding a document changes the digest"

    db.execute("UPDATE documents SET content_hash = 'hash-a2' WHERE source_path = 'x/a.md'")
    assert eval_retrieval.corpus_hash(db) != one, "re-ingesting under a new recipe changes it"


def test_eval_run_records_corpus_hash(db):
    db.execute(eval_retrieval.EVAL_SCHEMA_PATH.read_text())
    cols = {
        r[0] for r in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'eval_runs'"
        ).fetchall()
    }
    assert "corpus_hash" in cols
