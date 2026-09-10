"""OPM carrier letters: real, public, regulatory PDFs — OPM's numbered
guidance to FEHB/PSHB carriers (the annual call letter, technical guidance,
program announcements). The first external document family in the corpus.

The manifest is hand-curated in the sense the brochure list is: built once
by probing OPM's stable URL pattern (`carriers/{year}/{year}-{nn}.pdf`) for
the chosen years with the %PDF check, then fixed in the package (data/ is
not committed). `fetch` downloads by manifest through the OPM client
(browser User-Agent, magic-byte guard); `items` is the ingest discovery:
public tier, year = letter year, the letter's number, date, subject and URL
as record metadata. No PHI."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from raglab import config, opm
from raglab.metadata import DocumentMeta

MANIFEST_PATH = Path(__file__).parent / "manifests" / "carrier_letters.jsonl"
RAW_DIR = config.RAW_DIR / "carrier_letters"
URL = "https://www.opm.gov/healthcare-insurance/healthcare/carriers/{year}/{lid}.pdf"
YEARS = (2023, 2024, 2025, 2026)


@dataclass(frozen=True)
class Letter:
    letter_id: str
    year: int
    url: str
    date: str | None
    subject: str | None

    @property
    def pdf_path(self) -> Path:
        return RAW_DIR / str(self.year) / f"{self.letter_id}.pdf"

    @property
    def title(self) -> str:
        return f"OPM Carrier Letter {self.letter_id}" + (f": {self.subject}" if self.subject else "")


def load_manifest(path: Path = MANIFEST_PATH) -> list[Letter]:
    if not path.exists():
        return []
    return [Letter(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def first_page_fields(pdf: Path) -> tuple[str | None, str | None]:
    """(date, subject) from the letter's first page — OPM's fixed letterhead."""
    from pypdf import PdfReader

    text = " ".join(PdfReader(pdf).pages[0].extract_text().split())
    m_date = re.search(r"Date:\s*([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    m_subj = re.search(r"SUBJECT:\s*(.+)", text, re.I)
    subject = None
    if m_subj:
        subject = re.split(r"(?:\. |\bThis (?:Carrier )?[Ll]etter\b|\bThis is\b|\bThe purpose\b|\bLong recognized\b|\bIntroduction\b|\bSummary\b|\bBackground\b)", m_subj.group(1), maxsplit=1)[0]
        subject = subject.strip(" .:-")[:110] or None
    return (m_date.group(1) if m_date else None), subject


def build_manifest(years: tuple[int, ...] = YEARS, max_number: int = 40, path: Path = MANIFEST_PATH) -> list[Letter]:
    """Probe OPM for every letter of the years (numbers 1..max_number, stop
    after six consecutive misses past 10), download the PDFs, read date and
    subject from page one, and write the manifest. Run once; the manifest is
    then the fixed list."""
    import time

    letters = []
    with opm.client() as http:
        for year in years:
            misses = 0
            for n in range(1, max_number + 1):
                lid = f"{year}-{n:02d}"
                dest = RAW_DIR / str(year) / f"{lid}.pdf"
                if not (dest.exists() and dest.stat().st_size > 0):
                    result = _fetch(http, URL.format(year=year, lid=lid), dest)
                    if result.status != "fetched":
                        misses += 1
                        if misses >= 6 and n > 10:
                            break
                        time.sleep(0.2)
                        continue
                    misses = 0
                date, subject = first_page_fields(dest)
                letters.append(Letter(lid, year, URL.format(year=year, lid=lid), date, subject))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(l.__dict__) + "\n" for l in letters))
    return letters


def _fetch(http, url: str, dest: Path) -> opm.FetchResult:
    resp = http.get(url)
    if resp.status_code == 404:
        return opm.FetchResult("absent")
    if resp.status_code != 200:
        return opm.FetchResult("failed", f"HTTP {resp.status_code}")
    if not resp.content.startswith(b"%PDF"):
        return opm.FetchResult("failed", "not a PDF (magic bytes)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return opm.FetchResult("fetched", f"{len(resp.content) / 1e6:.1f} MB")


def fetch(letters: list[Letter] | None = None) -> dict[str, int]:
    """Download every manifest letter not already on disk."""
    counts = {"fetched": 0, "cached": 0, "absent": 0, "failed": 0}
    with opm.client() as http:
        for letter in letters or load_manifest():
            if letter.pdf_path.exists() and letter.pdf_path.stat().st_size > 0:
                counts["cached"] += 1
                continue
            counts[_fetch(http, letter.url, letter.pdf_path).status] += 1
    return counts


def meta(letter: Letter) -> DocumentMeta:
    return DocumentMeta(
        carrier="OPM", plan_code=None, plan_options=(), program="opm", year=letter.year,
        doc_type="carrier_letter", acl_tag="public",
        effective_date=letter.date or f"{letter.year}-01-01", title=letter.title,
        record={"letter_id": letter.letter_id, "letter_date": letter.date, "subject": letter.subject, "url": letter.url},
    )


def items() -> list[tuple[Path, DocumentMeta]]:
    """(pdf path, metadata) for every manifest letter present on disk."""
    return [(l.pdf_path, meta(l)) for l in load_manifest() if l.pdf_path.exists()]
