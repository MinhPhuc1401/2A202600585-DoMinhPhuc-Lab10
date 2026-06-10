from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from core.config import Settings
from core.utils import ensure_parent, normalize_whitespace


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


def _strip_html(text: str) -> str:
    """Remove HTML/XML tags from abstract text (Crossref wraps abstracts in JATS tags)."""
    return normalize_whitespace(re.sub(r"<[^>]+>", " ", text))


def _parse_date(date_parts: list) -> str:
    """Convert Crossref date-parts [[year, month, day]] to ISO string."""
    if not date_parts or not date_parts[0]:
        return ""
    parts = date_parts[0]
    year = parts[0] if len(parts) > 0 else 1970
    month = parts[1] if len(parts) > 1 else 1
    day = parts[2] if len(parts) > 2 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_pdf_url(item: dict) -> str:
    """Try to get a PDF link from the Crossref 'link' array."""
    for link in item.get("link", []):
        if link.get("content-type", "") == "application/pdf":
            return link.get("URL", "")
    links = item.get("link", [])
    return links[0].get("URL", "") if links else ""


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse Crossref payload into list PaperRecord.

    Steps:
    1. Iterate over payload["message"]["items"].
    2. Extract DOI, title, abstract, authors, subject, dates, URLs.
    3. Normalize text and skip invalid records.
    4. Return list[PaperRecord].
    """
    items = payload.get("message", {}).get("items", [])
    records: list[PaperRecord] = []

    for item in items:
        # --- paper_id (DOI) ---
        doi = item.get("DOI", "").strip()
        if not doi:
            continue

        # --- title ---
        title_list = item.get("title", [])
        title = normalize_whitespace(title_list[0]) if title_list else ""
        if not title:
            continue

        # --- abstract / summary ---
        raw_abstract = item.get("abstract", "")
        summary = _strip_html(raw_abstract).strip()
        if not summary:
            continue  # filter_source already has has-abstract:true, but guard anyway

        # --- authors ---
        authors: list[str] = []
        for author in item.get("author", []):
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        # --- categories / subjects ---
        categories: list[str] = [s.strip() for s in item.get("subject", []) if s.strip()]
        primary_category = categories[0] if categories else ""

        # --- dates ---
        published_parts = (
            item.get("published", {}).get("date-parts")
            or item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or []
        )
        published = _parse_date(published_parts)
        if not published:
            continue  # skip records with no parseable date

        updated_parts = item.get("deposited", {}).get("date-parts", [])
        updated = _parse_date(updated_parts)

        # --- URLs ---
        abs_url = f"https://doi.org/{doi}"
        pdf_url = _extract_pdf_url(item)

        records.append(
            PaperRecord(
                paper_id=doi,
                title=title,
                summary=summary,
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                abs_url=abs_url,
                pdf_url=pdf_url,
                comment="",
            )
        )

    return records


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Call Crossref API, save raw response, parse into records.

    Steps:
    1. Build params from settings (query, filter, rows).
    2. Call API with retry on 429/503.
    3. Save raw response to settings.paths.raw_api_response.
    4. Parse payload with parse_crossref_payload.
    5. Save records to settings.paths.raw_records_json.
    """
    base_url = "https://api.crossref.org/works"
    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
        "sort": "relevance",
        "select": (
            "DOI,title,abstract,author,subject,"
            "published,published-print,published-online,deposited,link"
        ),
    }

    # Retry logic for 429 / 503
    max_retries = 5
    backoff = 2.0
    payload: dict = {}
    for attempt in range(max_retries):
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code in (429, 503):
                wait = backoff * (2 ** attempt)
                print(f"[crossref] Rate limited ({resp.status_code}). Retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to fetch Crossref data after {max_retries} attempts: {exc}") from exc
            wait = backoff * (2 ** attempt)
            print(f"[crossref] Request error: {exc}. Retrying in {wait:.0f}s...")
            time.sleep(wait)

    # Save raw response
    ensure_parent(settings.paths.raw_api_response)
    settings.paths.raw_api_response.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[crossref] Raw response saved -> {settings.paths.raw_api_response}")

    # Parse
    records = parse_crossref_payload(payload)
    print(f"[crossref] Parsed {len(records)} valid records from {len(payload.get('message', {}).get('items', []))} items")

    # Save records
    ensure_parent(settings.paths.raw_records_json)
    settings.paths.raw_records_json.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[crossref] Records saved -> {settings.paths.raw_records_json}")

    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Load JSON snapshot and map to PaperRecord."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[PaperRecord] = []
    for item in data:
        records.append(
            PaperRecord(
                paper_id=item.get("paper_id", ""),
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                authors=item.get("authors", []),
                categories=item.get("categories", []),
                primary_category=item.get("primary_category", ""),
                published=item.get("published", ""),
                updated=item.get("updated", ""),
                abs_url=item.get("abs_url", ""),
                pdf_url=item.get("pdf_url", ""),
                comment=item.get("comment", ""),
            )
        )
    return records
