"""Identity, resolved at the edge (Phase 2 decision 3).

A username becomes an Identity: the document-lane persona, the warehouse
role, the granted surfaces, and an opaque user id. The MCP server (and the
Phase 4 login session) call `resolve` and hand the engine persona + role +
id; the engine stamps the id into the disclosure row and never reads it.
Policies stay role-only by construction.

Groups carry the entitlement. The seed is one group per row of the
entitlement matrix and one account per group; passwords are hashed with
stdlib scrypt (prototype-grade login, no dependency)."""

import hashlib
import os
import secrets
from dataclasses import dataclass, field

import psycopg

# group -> (persona, warehouse role, surfaces).
GROUPS: dict[str, tuple[str, str | None, tuple[str, ...], str]] = {
    "public":           ("public",          None,              ("ask",),                        "Unauthenticated / public benefits questions"),
    "call_center":      ("member_services", "MEMBER_SERVICES_REP", ("ask", "agent_assist"),      "Call-center representatives"),
    "appeals":          ("appeals",         "APPEALS_ANALYST",     ("ask", "appeals_workbench"), "Appeals analysts"),
    "benefits":         ("employee",        None,              ("ask",),                        "Benefits specialists"),
    "care_management":  ("care_team",       "CARE_MANAGER",    ("ask", "agent_assist"),         "Care managers"),
    "analytics":        ("public",          "ACTUARY",         ("ask", "analyst_view"),         "Analysts / actuaries"),
    "admin":            ("admin",           "CLAIMS_EXAMINER", ("ask", "agent_assist", "appeals_workbench", "analyst_view", "console"), "Platform administrators"),
}
# username -> (display name, group)
SEED_USERS: dict[str, tuple[str, str]] = {
    "rep.dana":     ("Dana Okafor",   "call_center"),
    "appeals.lee":  ("Morgan Lee",    "appeals"),
    "benefits.sam": ("Sam Whitcomb",  "benefits"),
    "cm.priya":     ("Priya Natarajan", "care_management"),
    "actuary.jo":   ("Jo Brennan",    "analytics"),
    "admin":        ("Platform Admin", "admin"),
}
DEMO_PASSWORD_ENV = "RAGLAB_DEMO_PASSWORD"  # one shared demo password (Phase 4 ruling: prototype-grade)


@dataclass(frozen=True)
class Identity:
    user_id: int
    username: str
    display_name: str
    group: str
    persona: str | None        # None = admin (owner connection, no SET ROLE)
    warehouse_role: str | None
    surfaces: tuple[str, ...] = field(default_factory=tuple)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return secrets.compare_digest(digest.hex(), digest_hex)


def seed(conn: psycopg.Connection, password: str | None = None) -> dict:
    """Groups, surface grants, and the six accounts. Idempotent; passwords
    are (re)set to the demo password (RAGLAB_DEMO_PASSWORD, default 'raglab-demo')."""
    password = password or os.environ.get(DEMO_PASSWORD_ENV, "raglab-demo")
    for name, (persona, role, surfaces, description) in GROUPS.items():
        conn.execute(
            "INSERT INTO groups (name, persona, warehouse_role, description) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET persona = EXCLUDED.persona, warehouse_role = EXCLUDED.warehouse_role, description = EXCLUDED.description",
            (name, persona, role, description),
        )
        conn.execute("DELETE FROM group_surfaces WHERE group_name = %s", (name,))
        for surface in surfaces:
            conn.execute("INSERT INTO group_surfaces (group_name, surface) VALUES (%s, %s)", (name, surface))
    for username, (display, group) in SEED_USERS.items():
        row = conn.execute(
            "INSERT INTO users (username, display_name, password_hash) VALUES (%s, %s, %s) "
            "ON CONFLICT (username) DO UPDATE SET display_name = EXCLUDED.display_name, password_hash = EXCLUDED.password_hash "
            "RETURNING id", (username, display, hash_password(password)),
        ).fetchone()
        conn.execute("DELETE FROM user_groups WHERE user_id = %s", (row[0],))
        conn.execute("INSERT INTO user_groups (user_id, group_name) VALUES (%s, %s)", (row[0], group))
    return {"groups": len(GROUPS), "users": len(SEED_USERS)}


def resolve(conn: psycopg.Connection, username: str) -> Identity:
    """username -> Identity. Unknown users are rejected (closed set); a user
    in several groups takes the first by group name (seed: exactly one)."""
    row = conn.execute(
        """SELECT u.id, u.username, u.display_name, g.name, g.persona, g.warehouse_role
           FROM users u JOIN user_groups ug ON ug.user_id = u.id JOIN groups g ON g.name = ug.group_name
           WHERE u.username = %s ORDER BY g.name LIMIT 1""", (username,)
    ).fetchone()
    if row is None:
        raise LookupError(f"unknown user {username!r}")
    surfaces = tuple(r[0] for r in conn.execute(
        "SELECT surface FROM group_surfaces WHERE group_name = %s ORDER BY surface", (row[3],)).fetchall())
    persona = None if row[4] == "admin" else row[4]
    return Identity(row[0], row[1], row[2], row[3], persona, row[5], surfaces)


def authenticate(conn: psycopg.Connection, username: str, password: str) -> Identity | None:
    row = conn.execute("SELECT password_hash FROM users WHERE username = %s", (username,)).fetchone()
    if row is None or not verify_password(password, row[0]):
        return None
    return resolve(conn, username)
