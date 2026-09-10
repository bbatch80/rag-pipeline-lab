"""RLS invariants (Phase 2, D13 reduced form) + fail-closed disclosure (D10).
Because policies are role-only, no user identity enters the engine; these
pin the properties that make that true."""

import pytest

from raglab import pipeline

pytestmark = pytest.mark.readonly

PERSONAS = ("persona_public", "persona_employee", "persona_care_team", "persona_member_services", "persona_appeals")
GOVERNED = ("documents", "chunks", "disclosure_log", "deid_vault", "appeal_evidence", "users", "groups", "user_groups")


def test_personas_hold_no_write_privilege_anywhere(db):
    rows = db.execute(
        "SELECT grantee, table_name, privilege_type FROM information_schema.role_table_grants "
        "WHERE grantee = ANY(%s) AND privilege_type <> 'SELECT'", (list(PERSONAS),)
    ).fetchall()
    assert rows == [], f"personas must be read-only consumers: {rows}"


def test_personas_cannot_read_the_vault_or_the_disclosure_log_or_identity(db):
    rows = db.execute(
        "SELECT grantee, table_name FROM information_schema.role_table_grants "
        "WHERE grantee = ANY(%s) AND table_name IN ('deid_vault', 'disclosure_log', 'users', 'user_groups', 'groups')",
        (list(PERSONAS),)
    ).fetchall()
    assert rows == [], rows
    for role in PERSONAS:
        db.execute(f"SET LOCAL ROLE {role}")
        for table in ("deid_vault", "disclosure_log", "users"):
            with pytest.raises(Exception):
                with db.transaction():
                    db.execute(f"SELECT count(*) FROM {table}")
        db.execute("RESET ROLE")


def test_no_security_definer_functions_and_no_views_over_governed_tables(db):
    assert db.execute(
        "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') AND p.prosecdef"
    ).fetchone()[0] == 0
    views = db.execute(
        "SELECT table_name FROM information_schema.views WHERE table_schema = 'public'"
    ).fetchall()
    assert views == [], f"no views over governed tables: {views}"


def test_unknown_persona_is_rejected():
    import psycopg

    with pytest.raises(ValueError):
        pipeline.run_query(None, "anything", persona="superuser")  # type: ignore[arg-type]


def test_disclosure_is_fail_closed(db, monkeypatch):
    """A failed disclosure insert means no payload, nothing committed, and an
    error that names the cause — not a retrieval failure."""
    monkeypatch.setattr("raglab.retrieval.embed_query", lambda t: "[" + ",".join(["0.5"] * 1536) + "]")

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "_disclose", boom)
    before = pipeline.DISCLOSURE_FAILURES["count"]
    with pytest.raises(RuntimeError, match="context withheld: disclosure record failed"):
        pipeline.run_query(db, "What is the HDHP deductible?", persona="public", source="test")
    assert pipeline.DISCLOSURE_FAILURES["count"] == before + 1
    # nothing landed: the fixture forbids commit, and the failed insert rolled the transaction back
    assert db.execute("SELECT count(*) FROM disclosure_log WHERE source = 'test'").fetchone()[0] == 0


class _NoCommit:
    """run_query commits; the fixture forbids it (tests never commit)."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def commit(self):
        pass


def test_disclosure_row_carries_the_user_id(db, monkeypatch):
    monkeypatch.setattr("raglab.retrieval.embed_query", lambda t: "[" + ",".join(["0.5"] * 1536) + "]")
    built = pipeline.run_query(_NoCommit(db), "What is the HDHP deductible?", persona="public", source="test", user_id=42)
    row = db.execute("SELECT user_id, persona FROM disclosure_log WHERE payload_id = %s", (built["payload_id"],)).fetchone()
    assert row == (42, "public")
