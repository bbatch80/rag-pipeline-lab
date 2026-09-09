"""Call-center shorthand: the vocabulary the call-note generator writes AND
the abbreviation dictionary the search copy expands (D8). One list, so the
dictionary covers the generator in full — a stated simplification: a real
dictionary is never complete, and coverage on real notes would be lower.
Ambiguous abbreviations (PT physical therapy / patient, OR, SO, MS) are
deliberately absent: an expansion that can be wrong is worse than none."""

import re

SHORTHAND: dict[str, str] = {
    "mbr": "member", "mbrs": "members", "pt": "patient", "dep": "dependent", "spouse": "spouse",
    "c/o": "complains of", "adv": "advised", "advd": "advised", "req": "requested", "rcvd": "received",
    "xfr": "transferred", "xfrd": "transferred", "cb": "call back", "f/u": "follow up",
    "re:": "regarding", "w/": "with", "w/o": "without", "b/c": "because",
    "ded": "deductible", "OOP": "out-of-pocket", "OOPM": "out-of-pocket maximum",
    "OON": "out-of-network", "INN": "in-network", "EOB": "explanation of benefits",
    "PA": "prior authorization", "auth": "authorization", "DOS": "date of service",
    "DOB": "date of birth", "ID#": "member ID", "PCP": "primary care provider",
    "spec": "specialist", "ER": "emergency room", "UC": "urgent care", "rx": "prescription",
    "hx": "history", "sx": "symptoms", "dx": "diagnosis", "tx": "treatment",
    "svc": "service", "svcs": "services", "appt": "appointment", "acct": "account",
    "ins": "insurance", "sec": "section", "amt": "amount", "bal": "balance", "pmt": "payment",
    "prem": "premium", "elig": "eligibility", "enr": "enrollment", "term": "terminated",
    "eff": "effective", "cov": "coverage", "covd": "covered", "info": "information",
    "ref": "referral", "sched": "scheduled", "conf": "confirmed", "adj": "adjusted",
    "OE": "open enrollment", "OS": "open season", "SPO": "Self Plus One", "S&F": "Self and Family",
    "HDHP": "high deductible health plan", "HSA": "health savings account",
    "CY": "calendar year", "YTD": "year to date", "N/A": "not applicable",
}
VERSION = "abbr:v1"  # part of the call-note processing recipe

_PATTERN = re.compile(
    r"(?<![\w/&])(" + "|".join(re.escape(k) for k in sorted(SHORTHAND, key=len, reverse=True)) + r")(?![\w/&])"
)


def expand(text: str) -> str:
    """Expand every whole-token abbreviation. Idempotent: expansions never
    contain a key as a whole token."""
    return _PATTERN.sub(lambda m: SHORTHAND[m.group(1)], text)
