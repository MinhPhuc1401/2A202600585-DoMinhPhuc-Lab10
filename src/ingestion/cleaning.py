from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from core.utils import compact_join, normalize_whitespace
from ingestion.crossref import PaperRecord


def _normalize_title(title: str) -> str:
    """Strip and title-case the paper title."""
    return normalize_whitespace(title).strip()


def _normalize_summary(summary: str) -> str:
    """Strip and normalize whitespace in abstract."""
    return normalize_whitespace(summary).strip()


def _parse_published(date_str: str) -> datetime | None:
    """Parse ISO date string to datetime. Returns None if invalid."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str[:len(fmt.replace("%Y", "0000").replace("%m", "00").replace("%d", "00"))], fmt)
        except ValueError:
            continue
    # Try partial match: extract year-month-day components
    match = re.match(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        day = int(match.group(3) or 1)
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Clean raw records into a DataFrame ready for embedding.

    Steps:
    1. Normalize title, summary, authors, categories.
    2. Parse published/updated date.
    3. Calculate age_days.
    4. Create helper columns: authors_joined, categories_joined, summary_chars, text_for_embedding.
    5. Drop duplicates and filter bad rows.
    6. Sort and return.
    """
    rows = []
    for record in records:
        title = _normalize_title(record.title)
        summary = _normalize_summary(record.summary)

        # Normalize authors: strip whitespace
        authors = [normalize_whitespace(a) for a in record.authors if a.strip()]
        # Normalize categories: strip whitespace
        categories = [normalize_whitespace(c) for c in record.categories if c.strip()]

        # Parse dates
        published_dt = _parse_published(record.published)
        updated_dt = _parse_published(record.updated)

        published_str = published_dt.strftime("%Y-%m-%d") if published_dt else ""
        updated_str = updated_dt.strftime("%Y-%m-%d") if updated_dt else ""

        # Calculate age_days (from run_date, strip timezone if needed)
        run_date_naive = run_date.replace(tzinfo=None) if run_date.tzinfo else run_date
        if published_dt:
            age_days = (run_date_naive - published_dt).days
        else:
            age_days = -1

        # Helper columns
        authors_joined = compact_join(authors, ", ")
        categories_joined = compact_join(categories, ", ")
        summary_chars = len(summary)

        # text_for_embedding: structured text combining key fields
        text_for_embedding = (
            f"Title: {title}\n"
            f"Authors: {authors_joined}\n"
            f"Categories: {categories_joined}\n"
            f"Summary: {summary}"
        )

        rows.append(
            {
                "paper_id": record.paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "primary_category": record.primary_category,
                "published": published_str,
                "updated": updated_str,
                "abs_url": record.abs_url,
                "pdf_url": record.pdf_url,
                "comment": record.comment,
                "age_days": age_days,
                "authors_joined": authors_joined,
                "categories_joined": categories_joined,
                "summary_chars": summary_chars,
                "text_for_embedding": text_for_embedding,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Drop duplicates by paper_id (keep first occurrence)
    df = df.drop_duplicates(subset=["paper_id"], keep="first")

    # Filter bad rows: empty title, or summary too short
    df = df[df["title"].str.strip().astype(bool)]
    df = df[df["summary_chars"] >= 20]

    # Filter rows with no valid published date
    df = df[df["published"].str.strip().astype(bool)]

    # Sort by published descending (newest first)
    df = df.sort_values("published", ascending=False).reset_index(drop=True)

    return df
