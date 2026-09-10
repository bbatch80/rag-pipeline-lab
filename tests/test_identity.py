"""Identity resolved at the edge (Phase 2 decision 3): username -> persona +
warehouse role + surfaces + opaque user id; closed set; scrypt passwords."""

import pytest

from raglab import identity

pytestmark = pytest.mark.readonly


def test_seed_is_idempotent_and_resolves_every_matrix_row(db):
    stats = identity.seed(db, password="pw")
    assert stats == {"groups": 7, "users": 6}
    assert identity.seed(db, password="pw") == stats  # re-run: same rows, no duplicates
    ident = identity.resolve(db, "rep.dana")
    assert (ident.group, ident.persona, ident.warehouse_role) == ("call_center", "member_services", "MEMBER_SERVICES_REP")
    assert identity.resolve(db, "appeals.lee").warehouse_role == "APPEALS_ANALYST"
    assert ident.surfaces == ("agent_assist", "ask") and ident.user_id > 0
    assert identity.resolve(db, "appeals.lee").persona == "appeals"
    assert identity.resolve(db, "cm.priya").persona == "care_team"
    assert identity.resolve(db, "actuary.jo").persona == "public" and identity.resolve(db, "actuary.jo").warehouse_role == "ACTUARY"
    admin = identity.resolve(db, "admin")
    assert admin.persona is None and "console" in admin.surfaces, "admin = owner connection, no SET ROLE"


def test_unknown_user_is_rejected_and_passwords_verify(db):
    identity.seed(db, password="pw")
    with pytest.raises(LookupError):
        identity.resolve(db, "nobody")
    assert identity.authenticate(db, "rep.dana", "pw").username == "rep.dana"
    assert identity.authenticate(db, "rep.dana", "wrong") is None
    assert identity.authenticate(db, "nobody", "pw") is None
    h = identity.hash_password("pw")
    assert h.startswith("scrypt$") and identity.verify_password("pw", h) and not identity.verify_password("PW", h)
    assert identity.hash_password("pw") != h, "fresh salt per hash"
