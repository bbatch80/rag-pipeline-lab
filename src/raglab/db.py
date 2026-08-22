"""Database connection helpers."""

import psycopg

from raglab import config


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or config.DATABASE_URL)
