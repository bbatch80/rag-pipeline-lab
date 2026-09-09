"""Near-duplicate marking for record sources (call notes): copy-paste
retries and double saves are whole-note near-duplicates. They get a
document row pointing at the original (duplicate_of) and no chunks, so
they never take a slot in retrieval but remain on disk and in the audit
path. Conservative on purpose: two genuine calls sharing macro text are
NOT duplicates — boilerplate suppression handles that; the threshold is
high (Jaccard >= 0.8 over word 3-shingles: a retried save adds one
line to a 40-60 word note and still clears it; two distinct calls that share
macro lines sit near 0.3-0.5)."""

import re

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
VERSION = "dedup:v2"  # part of record sources' processing recipe
THRESHOLD = 0.8
_lsh: dict[str, MinHashLSH] = {}
_WORD = re.compile(r"\w+")


def signature(text: str) -> MinHash:
    # The first line of a record is its header (record key, timestamp, rep):
    # a retried save gets a new key, so the header is excluded from the
    # comparison — content decides, not bookkeeping.
    body = text.split("\n", 1)[1] if "\n" in text else text
    words = _WORD.findall(body.lower())
    m = MinHash(num_perm=NUM_PERM)
    for i in range(max(1, len(words) - 2)):
        m.update(" ".join(words[i:i + 3]).encode())
    return m


def index_for(source_key: str) -> MinHashLSH:
    if source_key not in _lsh:
        _lsh[source_key] = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    return _lsh[source_key]


def seed(source_key: str, conn) -> int:
    """Load signatures of the source's already-ingested originals (one chunk
    each — record profile) so a later run still recognizes their copies."""
    lsh = index_for(source_key)
    n = 0
    rows = conn.execute(
        "SELECT d.id, c.content FROM documents d JOIN chunks c ON c.document_id = d.id "
        "JOIN sources s ON s.source_id = d.source_id WHERE s.key = %s AND d.duplicate_of IS NULL",
        (source_key,),
    ).fetchall()
    for doc_id, content in rows:
        key = f"doc:{doc_id}"
        if key not in lsh:
            lsh.insert(key, signature(content))
            n += 1
    return n


def check_and_add(source_key: str, doc_id: int, text: str) -> int | None:
    """Returns the original's document id if `text` duplicates one already
    seen for this source; otherwise records it and returns None."""
    lsh = index_for(source_key)
    sig = signature(text)
    hits = [h for h in lsh.query(sig) if h != f"doc:{doc_id}"]
    if hits:
        return int(hits[0].split(":")[1])
    lsh.insert(f"doc:{doc_id}", sig)
    return None


def forget(source_key: str, doc_id: int) -> None:
    """A re-ingested document gets a new id: drop the old key so copies
    seen later point at the row that exists."""
    lsh = index_for(source_key)
    key = f"doc:{doc_id}"
    if key in lsh:
        lsh.remove(key)


def reset():
    _lsh.clear()
