"""OPM brochure fetching. OPM 403s default HTTP clients (browser User-Agent
required) and can serve HTML error pages with 200s (magic-byte check required)."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE = "https://www.opm.gov/healthcare-insurance/healthcare/plan-information/plans"
PDF_URL = BASE + "/pdf/{year}/brochures/{ri}.pdf"
LISTING_URL = BASE + "/BrochureJson?brochureNumber={ri}&year={year}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


@dataclass
class FetchResult:
    status: str  # fetched | cached | absent | failed
    detail: str = ""


def client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True)


def fetch_listing(http: httpx.Client, ri: str, year: int, dest: Path) -> FetchResult:
    """Per-year existence probe. BrochureJson (despite the name) serves an HTML
    landing page: 404 = plan absent that year (data, not an error); 200 = the
    RI is live for that year. We record our own verification JSON."""
    if dest.exists():
        return FetchResult("cached")
    resp = http.get(LISTING_URL.format(ri=ri, year=year))
    if resp.status_code == 404:
        return FetchResult("absent", f"no listing for {ri}/{year}")
    if resp.status_code != 200:
        return FetchResult("failed", f"BrochureJson HTTP {resp.status_code}")
    title_match = re.search(r"<title>(.*?)</title>", resp.text, re.DOTALL)
    record = {
        "ri": ri,
        "year": year,
        "title": title_match.group(1).strip() if title_match else None,
        "verified_via": LISTING_URL.format(ri=ri, year=year),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2))
    return FetchResult("fetched")


def fetch_pdf(http: httpx.Client, ri: str, year: int, dest: Path) -> FetchResult:
    if dest.exists() and dest.stat().st_size > 0:
        return FetchResult("cached")
    resp = http.get(PDF_URL.format(ri=ri, year=year))
    if resp.status_code == 404:
        return FetchResult("absent", f"no brochure PDF for {ri}/{year}")
    if resp.status_code != 200:
        return FetchResult("failed", f"PDF HTTP {resp.status_code}")
    if not resp.content.startswith(b"%PDF"):
        return FetchResult("failed", "response is not a PDF (magic bytes)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return FetchResult("fetched", f"{len(resp.content) / 1e6:.1f} MB")
