"""Whole-database backup and restore, via pg_dump/pg_restore inside the
database container (the Mac has no Postgres client tools; the container
does). Files land in data/backups/ and never leave the machine.

A backup holds everything: corpus + embeddings + indexes, sources, the de-id
vault, the disclosure log, eval history, schema_migrations. Restore replaces
the live database with the file's contents."""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from raglab import config

CONTAINER = os.environ.get("RAGLAB_DB_CONTAINER", "raglab-db")
DB_USER = os.environ.get("RAGLAB_DB_USER", "raglab")
DB_NAME = os.environ.get("RAGLAB_DB_NAME", "raglab")
BACKUP_DIR = config.REPO_ROOT / "data" / "backups"


def _docker(*args: str, stdin=None, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-i", CONTAINER, *args],
        stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, check=True,
    )


def create(tag: str = "") -> Path:
    """pg_dump in custom format (compressed, restorable table by table)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"raglab-{stamp}" + (f"-{tag}" if tag else "") + ".dump"
    path = BACKUP_DIR / name
    with path.open("wb") as out:
        _docker("pg_dump", "-U", DB_USER, "-Fc", "--no-owner", DB_NAME, stdout=out)
    return path


def inventory(path: Path) -> list[str]:
    """Table names the file contains (pg_restore --list; reads the file only)."""
    with path.open("rb") as src:
        result = _docker("pg_restore", "--list", stdin=src, stdout=subprocess.PIPE)
    tables = []
    for line in result.stdout.decode().splitlines():
        parts = line.split()
        # "3746; 0 190292 TABLE DATA public chunks raglab"
        if len(parts) >= 7 and parts[3] == "TABLE" and parts[4] == "DATA":
            tables.append(parts[6])
    return sorted(tables)


def restore(path: Path) -> None:
    """Replace the live database with the file. Existing objects are dropped
    first (--clean --if-exists); roles live at the cluster level and survive."""
    with path.open("rb") as src:
        _docker(
            "pg_restore", "-U", DB_USER, "-d", DB_NAME,
            "--clean", "--if-exists", "--no-owner", "--exit-on-error",
            stdin=src, stdout=subprocess.DEVNULL,
        )


def available() -> list[Path]:
    if not BACKUP_DIR.exists():
        return []
    return sorted(BACKUP_DIR.glob("raglab-*.dump"))
