"""Backup files are complete and readable. Runs only where the database
container is reachable (skipped in CI); never restores into the live DB."""

import shutil
import subprocess

import pytest

from raglab import backup

docker = shutil.which("docker")
pytestmark = pytest.mark.skipif(
    docker is None
    or subprocess.run([docker, "inspect", backup.CONTAINER], capture_output=True).returncode != 0,
    reason="database container not reachable",
)


def test_backup_contains_every_governed_table(tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "BACKUP_DIR", tmp_path)
    path = backup.create("test")
    assert path.exists() and path.stat().st_size > 10_000  # CI's database is empty
    tables = set(backup.inventory(path))
    assert {"documents", "chunks", "sources", "deid_vault", "disclosure_log",
            "eval_runs", "eval_scores", "schema_migrations"} <= tables
