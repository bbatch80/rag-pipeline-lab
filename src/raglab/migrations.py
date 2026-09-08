"""Additive schema migrations: numbered SQL files applied once, in order.

`db/schema.sql` is the destructive base (drop-and-recreate) for a fresh
database. Everything after it is a migration, so the corpus survives every
schema change. Each file runs in its own transaction and is recorded in
`schema_migrations`; applying again is a no-op."""

import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from raglab import config

_FILENAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path


def available(directory: Path = config.MIGRATIONS_DIR) -> list[Migration]:
    found = []
    for path in sorted(directory.glob("*.sql")):
        m = _FILENAME.match(path.name)
        if not m:
            raise ValueError(f"migration filename must be NNN_name.sql: {path.name}")
        found.append(Migration(int(m.group(1)), m.group(2), path))
    versions = [m.version for m in found]
    if len(set(versions)) != len(versions):
        raise ValueError(f"duplicate migration versions: {versions}")
    return found


def applied(conn: psycopg.Connection) -> set[int]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version integer PRIMARY KEY, name text NOT NULL,"
        " applied_at timestamptz NOT NULL DEFAULT now())"
    )
    return {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}


def apply(conn: psycopg.Connection, directory: Path = config.MIGRATIONS_DIR) -> list[Migration]:
    """Apply every migration not yet recorded. Returns the ones applied.

    Never commits the caller's transaction: each migration runs inside
    `conn.transaction()`, a savepoint when a transaction is already open
    (tests) and a real transaction otherwise (the CLI)."""
    done = applied(conn)
    ran = []
    for migration in available(directory):
        if migration.version in done:
            continue
        with conn.transaction():
            conn.execute(migration.path.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                (migration.version, migration.name),
            )
        ran.append(migration)
    return ran
