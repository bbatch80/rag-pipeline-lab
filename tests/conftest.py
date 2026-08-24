import pytest

from raglab import db as raglab_db


@pytest.fixture
def db():
    """Connection whose work is always rolled back — tests leave no residue.
    Tables are emptied inside the transaction so tests see a clean slate
    regardless of what the live database holds; rollback restores it."""
    with raglab_db.connect() as conn:
        conn.execute("DELETE FROM quarantine")
        conn.execute("DELETE FROM documents")
        yield conn
        conn.rollback()
