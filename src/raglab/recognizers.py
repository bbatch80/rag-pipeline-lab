"""Custom Presidio recognizers for the identifiers this platform governs.

Stock Presidio has no notion of a member ID, an MRN, a claim number, or an
appeal case number, and it misses shorthand dates and member names written
without context. Each recognizer here accepts every surface form a rep or
clinician types, and the check-digit identifiers are validated so lookalike
numbers are not flagged. Member names come from the enrollment roster: a
payer knows its members (entity-anchored recognition; provider and family
names stay with the statistical recognizer)."""

import re

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

from raglab import identifiers

# Entity names as Presidio (and the vault) see them.
MEMBER_ID, MRN, CLAIM_ID, CASE_ID = "MEMBER_ID", "MRN", "CLAIM_ID", "CASE_ID"
IDENTIFIER_KIND = {MEMBER_ID: "member_id", MRN: "mrn", CLAIM_ID: "claim_id", CASE_ID: "case_id"}


class _Checked(PatternRecognizer):
    """A pattern recognizer whose matches must canonicalize (check digit)."""

    def __init__(self, entity: str, patterns: list[Pattern], context: list[str]):
        super().__init__(supported_entity=entity, patterns=patterns, context=context,
                         supported_language="en", name=f"raglab_{entity.lower()}")
        self._kind = IDENTIFIER_KIND[entity]

    def validate_result(self, pattern_text: str):
        return identifiers.canonicalize(self._kind, pattern_text) is not None


def member_id_recognizer() -> PatternRecognizer:
    return _Checked(MEMBER_ID, [
        Pattern("member_prefixed", r"\bM[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}\b", 0.7),
        Pattern("member_bare", r"\b\d{9}\b", 0.3),  # needs context or it is just a number
    ], context=["member", "member id", "id#", "subscriber", "mbr", "enrollee"])


def mrn_recognizer() -> PatternRecognizer:
    return PatternRecognizer(supported_entity=MRN, name="raglab_mrn", supported_language="en",
                             patterns=[Pattern("mrn", r"\bMRN[\s#:-]*\d{7}\b", 0.8)],
                             context=["mrn", "medical record", "record number", "chart"])


def claim_id_recognizer() -> PatternRecognizer:
    return _Checked(CLAIM_ID, [
        Pattern("claim", r"\bCLM[\s-]?(?:\d{10}|\d{3}[\s-]?\d{3}[\s-]?\d{4})\b", 0.8),
    ], context=["claim", "claim number", "claim #", "clm"])


def case_id_recognizer() -> PatternRecognizer:
    return _Checked(CASE_ID, [Pattern("case", r"\bAPL[\s-]?\d{7}\b", 0.8)],
                    context=["appeal", "case", "case number", "grievance"])


_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"


def shorthand_date_recognizer() -> PatternRecognizer:
    """The forms reps and clinicians actually write: 2/11/24, 02.11.2024,
    2-11-2024, Feb 11, Feb 11 2024. Emits DATE_TIME beside spaCy's."""
    return PatternRecognizer(supported_entity="DATE_TIME", name="raglab_shorthand_date", supported_language="en",
                             patterns=[
                                 Pattern("numeric", r"\b\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})\b", 0.6),
                                 Pattern("iso", r"\b\d{4}-\d{2}-\d{2}\b", 0.6),
                                 Pattern("month_day", rf"\b(?:{_MONTHS})[a-z]*\.? \d{{1,2}}(?:,? \d{{4}})?\b", 0.6),
                             ])


class RosterNameRecognizer(EntityRecognizer):
    """Member names from the enrollment roster, as PERSON. Full names in
    either order score high; a bare first or last name scores lower and
    relies on context words nearby (pt, patient, member) via Presidio's
    context enhancement."""

    def __init__(self, first_names: set[str], last_names: set[str], pairs: set[tuple[str, str]]):
        super().__init__(supported_entities=["PERSON"], supported_language="en", name="raglab_roster_names",
                         context=["pt", "patient", "member", "mbr", "name", "enrollee", "spouse", "dependent"])
        self.first, self.last, self.pairs = first_names, last_names, pairs
        self._token = re.compile(r"\b[A-Z][a-z]{2,}(?:['-][A-Z][a-z]+)?\b")

    def load(self):
        pass

    def _result(self, start: int, end: int, score: float) -> RecognizerResult:
        return RecognizerResult("PERSON", start, end, score,
                                analysis_explanation=AnalysisExplanation(recognizer=self.name, original_score=score))

    def analyze(self, text, entities, nlp_artifacts=None):
        if "PERSON" not in entities:
            return []
        out, tokens = [], list(self._token.finditer(text))
        used = set()
        for i, m in enumerate(tokens):
            if i in used:
                continue
            w = m.group(0)
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            # "First Last"
            if nxt and (w, nxt.group(0)) in self.pairs and nxt.start() - m.end() <= 2:
                out.append(self._result(m.start(), nxt.end(), 0.85))
                used.update({i, i + 1}); continue
            # "Last, First"
            if nxt and (nxt.group(0), w) in self.pairs and text[m.end():nxt.start()] in (", ", ","):
                out.append(self._result(m.start(), nxt.end(), 0.85))
                used.update({i, i + 1}); continue
            if w in self.first or w in self.last:
                out.append(self._result(m.start(), m.end(), 0.45))
        return out


def roster_from_db(conn) -> RosterNameRecognizer:
    rows = conn.execute(
        "SELECT first, last FROM synthea.patients WHERE first IS NOT NULL AND last IS NOT NULL"
    ).fetchall()
    from raglab.synth.notes import clean_name
    pairs = {(clean_name(f), clean_name(l)) for f, l in rows}
    return RosterNameRecognizer({f for f, _ in pairs}, {l for _, l in pairs}, pairs)


def phone_pattern_recognizer() -> PatternRecognizer:
    """North-American phone shapes by pattern, beside Presidio's validating
    recognizer: a typo'd or unassigned area code is still a phone number a
    rep wrote down, and a leaked one is still a leak."""
    return PatternRecognizer(supported_entity="PHONE_NUMBER", name="raglab_phone_shape", supported_language="en",
                             patterns=[Pattern("nanp", r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b", 0.5)],
                             context=["phone", "call", "contact", "tel", "cell", "fax", "office"])


def all_recognizers(conn=None) -> list:
    recs = [member_id_recognizer(), mrn_recognizer(), claim_id_recognizer(),
            case_id_recognizer(), shorthand_date_recognizer(), phone_pattern_recognizer()]
    if conn is not None:
        try:
            recs.append(roster_from_db(conn))
        except Exception:  # no synthea schema (CI): names stay statistical
            conn.rollback() if hasattr(conn, "rollback") else None
    return recs
