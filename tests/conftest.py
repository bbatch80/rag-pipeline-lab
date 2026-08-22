import pytest

from raglab import db as raglab_db


@pytest.fixture
def db():
    """Connection whose work is always rolled back — tests leave no residue."""
    with raglab_db.connect() as conn:
        yield conn
        conn.rollback()
