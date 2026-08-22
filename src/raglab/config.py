"""Central configuration. All environment access lives here."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"

load_dotenv(REPO_ROOT / ".env")

DATABASE_URL = os.environ.get(
    "RAGLAB_DATABASE_URL",
    "postgresql://raglab:raglab@localhost:5433/raglab",
)

EMBEDDING_DIMENSIONS = 1536
