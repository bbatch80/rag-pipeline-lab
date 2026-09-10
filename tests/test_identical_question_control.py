"""Phase 2 exit criterion: the identical-question control. One question, run
through the real pipeline (RLS, member context, reranker, disclosure) as every
vector persona, returns exactly the tier map — nothing more for any role,
nothing less for the entitled one. Golden persona items are pairwise; this is
the five-way assertion, kept in CI as a test."""
import psycopg
import pytest

from raglab.pipeline import PERSONAS, run_query
from test_governance import VISIBILITY, _seed_tiers  # tests/ is on sys.path under pytest's prepend import mode

# One source of truth for who sees what: the RLS matrix test's table (care_team
# is a lateral tier walled from employee material both ways; the two Phase 2
# operations tiers inherit employee).
TIER_MAP = {role.removeprefix("persona_"): set(tags) for role, tags in VISIBILITY.items()}
MEMBER_SCOPED = {"care_team", "member_services", "appeals"}  # records about one person
FAKE_MEMBER = "M999900004"  # passes the shape + Luhn check
FAKE_PATIENT = "00000000-0000-4000-8000-000000000001"


class _NoCommit:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, *a, **k):
        return self._conn.execute(*a, **k)

    def commit(self):
        pass

    def transaction(self):  # resolve_context opens a savepoint for each id lookup
        return self._conn.transaction()


def _exact_scan(db):
    # The fixture empties the tables inside a transaction, so the HNSW index is
    # mostly invisible rows: a bounded candidate walk misses the few live ones at
    # random. Exact scan makes the vector lane deterministic (as in the floor test).
    db.execute("SET LOCAL enable_indexscan = off")


def _fake_models(monkeypatch):
    monkeypatch.setattr("raglab.retrieval.embed_query", lambda text: "[" + ",".join(["0.5"] * 1536) + "]")

    class _FakeModel:
        def predict(self, pairs):
            return [0.9] * len(pairs)

    import raglab.rerank as rr

    monkeypatch.setattr(rr, "_model", _FakeModel())


def _tags(payload):
    return {c["source"]["title"].removeprefix("doc-") for c in payload.get("chunks", [])}


def test_same_question_every_persona_without_a_member_returns_the_tier_map_minus_person_records(db, monkeypatch):
    _seed_tiers(db, per_tier=1, embed=True)  # one chunk per tier: every visible tier fits in the payload even when the fake reranker ties
    _fake_models(monkeypatch)
    _exact_scan(db)
    for persona in PERSONAS:
        got = _tags(run_query(_NoCommit(db), "what do the secret facts say", persona=persona))
        db.execute("RESET ROLE")  # run_query's SET LOCAL ROLE normally ends with its commit; the proxy never commits
        assert got == TIER_MAP[persona] - MEMBER_SCOPED, persona


def test_same_question_every_persona_with_a_member_returns_exactly_the_tier_map(db, monkeypatch):
    _seed_tiers(db, per_tier=1, embed=True)  # one chunk per tier: every visible tier fits in the payload even when the fake reranker ties
    _fake_models(monkeypatch)
    _exact_scan(db)
    try:
        with db.transaction():
            db.execute("INSERT INTO synthea.patients (id, member_id) VALUES (%s, %s)", (FAKE_PATIENT, FAKE_MEMBER))
    except (psycopg.errors.UndefinedTable, psycopg.errors.NotNullViolation) as exc:
        pytest.skip(f"no usable synthea.patients here: {exc.__class__.__name__}")
    db.execute("UPDATE documents SET member_key = %s WHERE title LIKE 'doc-%%'", (FAKE_PATIENT,))
    db.execute("UPDATE chunks SET member_key = %s WHERE document_id IN (SELECT id FROM documents WHERE title LIKE 'doc-%%')", (FAKE_PATIENT,))
    for persona in PERSONAS:
        got = _tags(run_query(_NoCommit(db), f"what do the secret facts say about member {FAKE_MEMBER}",
                              persona=persona, member_id=FAKE_MEMBER))
        db.execute("RESET ROLE")
        assert got == TIER_MAP[persona], persona
