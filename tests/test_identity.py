"""Identity resolved at the edge (Phase 2 decision 3): username -> persona +
warehouse role + surfaces + opaque user id; closed set; scrypt passwords."""

import pytest

from raglab import identity

pytestmark = pytest.mark.readonly


def test_seed_is_idempotent_and_resolves_every_matrix_row(db):
    stats = identity.seed(db, password="pw")
    assert stats == {"groups": 7, "users": 6}
    assert identity.seed(db, password="pw") == stats  # re-run: same rows, no duplicates
    ident = identity.resolve(db, "member_services")
    assert (ident.group, ident.persona, ident.warehouse_role) == ("call_center", "member_services", "MEMBER_SERVICES_REP")
    assert identity.resolve(db, "appeals").warehouse_role == "APPEALS_ANALYST"
    assert ident.surfaces == ("agent_assist", "ask") and ident.user_id > 0
    assert identity.resolve(db, "appeals").persona == "appeals"
    assert identity.resolve(db, "care_team").persona == "care_team"
    assert identity.resolve(db, "analyst").persona == "public" and identity.resolve(db, "analyst").warehouse_role == "ACTUARY"
    admin = identity.resolve(db, "admin")
    assert admin.persona is None and "console" in admin.surfaces, "admin = owner connection, no SET ROLE"


def test_unknown_user_is_rejected_and_passwords_verify(db):
    identity.seed(db, password="pw")
    with pytest.raises(LookupError):
        identity.resolve(db, "nobody")
    assert identity.authenticate(db, "member_services", "pw").username == "member_services"
    assert identity.authenticate(db, "member_services", "wrong") is None
    assert identity.authenticate(db, "nobody", "pw") is None
    h = identity.hash_password("pw")
    assert h.startswith("scrypt$") and identity.verify_password("pw", h) and not identity.verify_password("PW", h)
    assert identity.hash_password("pw") != h, "fresh salt per hash"


def test_a_retired_named_account_is_re_attributed_to_its_role(db):
    """Logins are roles, not people (2026-09-15): an old named account's
    disclosure history moves to the role that replaced it and the account
    goes, so no invented person appears anywhere on the site."""
    identity.seed(db)
    old = db.execute("INSERT INTO users (username, display_name, password_hash) VALUES ('rep.dana', 'Dana Okafor', 'x') RETURNING id").fetchone()[0]
    db.execute("INSERT INTO disclosure_log (payload_id, persona, source, query, payload_status, chunk_ids, content_hashes, doc_titles, acl_basis, payload, user_id) "
               "VALUES (gen_random_uuid(), 'member_services', 'web', 'q', 'ok', '{}', '{}', '{}', '{}', '{}'::jsonb, %s)", (old,))
    identity.seed(db)
    assert db.execute("SELECT count(*) FROM users WHERE username = 'rep.dana'").fetchone()[0] == 0
    new_id = db.execute("SELECT id FROM users WHERE username = 'member_services'").fetchone()[0]
    assert db.execute("SELECT user_id FROM disclosure_log WHERE query = 'q'").fetchone()[0] == new_id
