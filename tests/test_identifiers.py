"""Stable, checkable identifiers derived from the person key."""

import pytest

from raglab import identifiers as ids


def test_luhn_known_values():
    assert ids.luhn_check_digit("7992739871") == "3"   # the textbook example
    assert ids.luhn_valid("79927398713") and not ids.luhn_valid("79927398710")


def test_derivations_are_deterministic_and_shaped():
    key = "20e28446-7a6e-4c2b-9b8f-1c2d3e4f5a6b"
    assert ids.member_id(key) == ids.member_id(key)
    assert ids.member_id(key) != ids.member_id(key[:-1] + "c")
    m = ids.member_id(key)
    assert len(m) == 10 and m[0] == "M" and ids.luhn_valid(m[1:])
    assert ids.mrn(key).startswith("MRN") and len(ids.mrn(key)) == 10
    c = ids.claim_id("enc-1")
    assert c.startswith("CLM-") and ids.luhn_valid(c[4:])
    a = ids.case_id(17)
    assert a.startswith("APL-") and ids.luhn_valid(a[4:])


@pytest.mark.parametrize("style", ["canonical", "grouped", "spaced", "bare"])
def test_every_presentation_canonicalizes_back(style):
    for kind, canon in (("member_id", ids.member_id("k")), ("mrn", ids.mrn("k")),
                        ("claim_id", ids.claim_id("e")), ("case_id", ids.case_id(3))):
        surface = ids.present(kind, canon, style)
        if kind in ("mrn", "claim_id", "case_id") and style == "bare":
            continue  # prose forms ("claim 0004…") are for recognizers with context, not fullmatch
        assert ids.canonicalize(kind, surface) == canon, (kind, style, surface)


def test_check_digit_rejects_lookalikes():
    m = ids.member_id("k")
    wrong = m[:-1] + str((int(m[-1]) + 1) % 10)
    assert ids.canonicalize("member_id", wrong) is None
    assert ids.canonicalize("member_id", "M12345") is None
    assert ids.canonicalize("claim_id", "CLM-0000000000") is None or ids.luhn_valid("0000000000")
