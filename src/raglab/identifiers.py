"""Stable member identifiers, derived — never invented — from the person key.

Formats (Phase 1 decision 3):
  member ID   M + 8 digits + Luhn check          M048217336
  MRN         MRN + 7 digits (no check digit)     MRN4408125
  claim ID    CLM- + 9 digits + Luhn check         CLM-0004183742
  case ID     APL- + 6 digits + Luhn check         APL-0038174

Every value is a deterministic function of (project seed, source key). Text
may present a value with separators or spaces; canonicalize() maps every
presentation back to the one canonical string, verifying the check digit,
so the vault issues exactly one pseudonym per entity across all sources."""

import hashlib
import re

SEED = "raglab-v2"


def _digits(kind: str, key: str, n: int) -> str:
    h = hashlib.sha256(f"{SEED}|{kind}|{key}".encode()).hexdigest()
    return str(int(h, 16) % 10**n).zfill(n)


def luhn_check_digit(body: str) -> str:
    total, double = 0, True  # rightmost body digit is doubled (check digit appended after)
    for ch in reversed(body):
        d = int(ch)
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return str((10 - total % 10) % 10)


def luhn_valid(digits: str) -> bool:
    return len(digits) > 1 and digits.isdigit() and luhn_check_digit(digits[:-1]) == digits[-1]


def member_id(person_key: str, attempt: int = 0) -> str:
    """attempt > 0 only when assignment hit a collision (see enrollment.assign);
    the table is the authority for the value a person actually carries."""
    body = _digits("member", f"{person_key}#{attempt}" if attempt else person_key, 8)
    return f"M{body}{luhn_check_digit(body)}"


def mrn(person_key: str, attempt: int = 0) -> str:
    return f"MRN{_digits('mrn', f'{person_key}#{attempt}' if attempt else person_key, 7)}"


def claim_id(encounter_key: str) -> str:
    body = _digits("claim", encounter_key, 9)
    return f"CLM-{body}{luhn_check_digit(body)}"


def case_id(sequence: int) -> str:
    body = _digits("case", str(sequence), 6)
    return f"APL-{body}{luhn_check_digit(body)}"


# Surface forms a rep or clinician might type. The recognizers accept these;
# canonicalize() collapses them.
_MEMBER = re.compile(r"(?i)\bM[\s-]?(\d{3})[\s-]?(\d{3})[\s-]?(\d{3})\b")
_MEMBER_BARE = re.compile(r"\b(\d{9})\b")
_MRN = re.compile(r"(?i)\bMRN[\s#:-]*(\d{7})\b")
_CLAIM = re.compile(r"(?i)\bCLM[\s-]?(?:(\d{10})|(\d{3})[\s-]?(\d{3})[\s-]?(\d{4}))\b")
_CASE = re.compile(r"(?i)\bAPL[\s-]?(\d{7})\b")


def canonicalize(kind: str, surface: str) -> str | None:
    """Canonical value for a surface form of `kind`, or None if it is not a
    valid identifier of that kind (wrong shape or failed check digit)."""
    s = surface.strip()
    if kind == "member_id":
        m = _MEMBER.fullmatch(s) or _MEMBER_BARE.fullmatch(s)
        if not m:
            return None
        digits = "".join(g for g in m.groups() if g)
        return f"M{digits}" if luhn_valid(digits) else None
    if kind == "mrn":
        m = _MRN.fullmatch(s)
        return f"MRN{m.group(1)}" if m else None
    if kind == "claim_id":
        m = _CLAIM.fullmatch(s)
        if not m:
            return None
        digits = "".join(g for g in m.groups() if g)
        return f"CLM-{digits}" if luhn_valid(digits) else None
    if kind == "case_id":
        m = _CASE.fullmatch(s)
        return f"APL-{m.group(1)}" if m and luhn_valid(m.group(1)) else None
    raise ValueError(kind)


def present(kind: str, canonical: str, style: str) -> str:
    """A surface form of a canonical identifier, for generators. Styles:
    canonical | grouped | spaced | bare (member ID only, digits alone)."""
    if kind == "member_id":
        d = canonical[1:]
        return {"canonical": canonical, "grouped": f"M{d[:3]}-{d[3:6]}-{d[6:]}",
                "spaced": f"M {d}", "bare": d}[style]
    if kind == "mrn":
        return {"canonical": canonical, "grouped": canonical, "spaced": f"MRN {canonical[3:]}",
                "bare": f"MRN# {canonical[3:]}"}[style]
    if kind == "claim_id":
        d = canonical[4:]
        return {"canonical": canonical, "grouped": f"CLM-{d[:3]}-{d[3:6]}-{d[6:]}",
                "spaced": f"CLM {d}", "bare": f"claim {d}"}[style]
    if kind == "case_id":
        return {"canonical": canonical, "grouped": canonical, "spaced": f"APL {canonical[4:]}",
                "bare": f"case {canonical[4:]}"}[style]
    raise ValueError(kind)


_SURFACE = {
    "member_id": re.compile(r"(?i)\bM[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}\b"),
    "mrn": re.compile(r"(?i)\bMRN[\s#:-]*\d{7}\b"),
    "claim_id": re.compile(r"(?i)\bCLM[\s-]?(?:\d{10}|\d{3}[\s-]?\d{3}[\s-]?\d{4})\b"),
    "case_id": re.compile(r"(?i)\bAPL[\s-]?\d{7}\b"),
}


def normalize_identifiers(text: str) -> str:
    """Rewrite every recognizable identifier surface form in text to its
    canonical form (valid check digits only). Used on queries before vault
    translation and on the search copy of normalized sources."""
    out = text
    for kind, pattern in _SURFACE.items():
        def _sub(m, kind=kind):
            canon = canonicalize(kind, m.group(0))
            return canon if canon else m.group(0)
        out = pattern.sub(_sub, out)
    return out
