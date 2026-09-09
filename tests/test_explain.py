"""raglab explain-golden: the three facts behind a hit or a miss."""

import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="needs the embedded corpus and the embedder (not in CI's fresh database)"
)]


def test_explain_reports_pool_and_rerank_facts():
    from raglab import db, explain

    with db.connect() as conn:
        ex = explain.explain(conn, "C7")
        conn.rollback()
    assert ex.member_key and ex.pool_size > 0
    assert ex.expected and all("in_pool" in e and "rerank_position" in e for e in ex.expected)
    assert len(ex.top) <= 10 and ex.verdict[1] >= 0.0
