"""OPM carrier letters: fixed manifest, fetch guard, ingest metadata."""

import pytest

from raglab import letters


def test_manifest_is_fixed_and_well_formed():
    ls = letters.load_manifest()
    assert len(ls) == 73 and {l.year for l in ls} == set(letters.YEARS)
    assert all(l.url.endswith(f"/{l.year}/{l.letter_id}.pdf") and l.letter_id.startswith(str(l.year)) for l in ls)
    assert all(l.subject for l in ls), "every letter carries its SUBJECT line"
    assert len({l.letter_id for l in ls}) == 73


def test_meta_is_public_dated_and_carries_the_record():
    l = letters.load_manifest()[0]
    m = letters.meta(l)
    assert m.acl_tag == "public" and m.doc_type == "carrier_letter" and m.year == l.year
    assert m.title.startswith(f"OPM Carrier Letter {l.letter_id}: ")
    assert m.record["letter_id"] == l.letter_id and m.record["url"] == l.url and m.plan_code is None


def test_items_only_lists_letters_present_on_disk():
    present = letters.items()
    assert all(p.exists() for p, _ in present)
    if not present:
        pytest.skip("letters not downloaded on this machine (CI)")
    assert len(present) == 73


@pytest.mark.skipif(not letters.RAW_DIR.exists(), reason="letters not downloaded on this machine (CI)")
def test_first_page_fields_reads_opm_letterhead():
    l = next(x for x in letters.load_manifest() if x.letter_id == "2025-01")
    date, subject = letters.first_page_fields(l.pdf_path)
    assert date == "January 15, 2025" and subject and "Call Letter" in subject
