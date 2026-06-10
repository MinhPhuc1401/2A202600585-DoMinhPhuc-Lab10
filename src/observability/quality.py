from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings
from core.utils import ensure_parent


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Run data quality checks and write report to data/quality/.

    Steps:
    1. Check row count > 0.
    2. Check paper_id not null and unique.
    3. Check title not null.
    4. Check summary length >= 20 chars.
    5. Check freshness: age_days <= freshness_threshold_days.
    6. Write result JSON to data/quality/{report_name}.json.
    """
    checks: list[dict[str, Any]] = []
    total_rows = len(df)

    # --- Check 1: Row count ---
    row_count_ok = total_rows > 0
    checks.append({
        "name": "row_count_gt_zero",
        "passed": row_count_ok,
        "detail": f"Total rows: {total_rows}",
    })

    if total_rows == 0:
        # No point in running further checks on empty dataframe
        report = {
            "report_name": report_name,
            "total_rows": total_rows,
            "checks": checks,
            "passed": 0,
            "failed": len(checks),
            "success_rate": 0.0,
        }
        _save_report(report, settings, report_name)
        return report

    # --- Check 2: paper_id not null ---
    null_id_count = df["paper_id"].isna().sum() + (df["paper_id"].astype(str).str.strip() == "").sum()
    checks.append({
        "name": "paper_id_not_null",
        "passed": int(null_id_count) == 0,
        "detail": f"Null/empty paper_id count: {null_id_count}",
    })

    # --- Check 3: paper_id unique ---
    dup_id_count = int(df["paper_id"].duplicated().sum())
    checks.append({
        "name": "paper_id_unique",
        "passed": dup_id_count == 0,
        "detail": f"Duplicate paper_id count: {dup_id_count}",
    })

    # --- Check 4: title not null ---
    null_title_count = df["title"].isna().sum() + (df["title"].astype(str).str.strip() == "").sum()
    checks.append({
        "name": "title_not_null",
        "passed": int(null_title_count) == 0,
        "detail": f"Null/empty title count: {null_title_count}",
    })

    # --- Check 5: summary length >= 20 chars ---
    short_summary_count = int((df["summary_chars"] < 20).sum())
    checks.append({
        "name": "summary_min_length_20",
        "passed": short_summary_count == 0,
        "detail": f"Rows with summary < 20 chars: {short_summary_count}",
    })

    # --- Check 6: freshness (age_days <= threshold) ---
    threshold = settings.freshness_threshold_days
    stale_count = int((df["age_days"] > threshold).sum())
    stale_pct = round(stale_count / total_rows * 100, 1) if total_rows > 0 else 0.0
    checks.append({
        "name": f"freshness_age_lte_{threshold}_days",
        "passed": stale_count == 0,
        "detail": f"Stale rows (age > {threshold} days): {stale_count} ({stale_pct}%)",
    })

    # Summary
    passed = sum(1 for c in checks if c["passed"])
    failed = len(checks) - passed
    success_rate = round(passed / len(checks), 4) if checks else 0.0

    report: dict[str, Any] = {
        "report_name": report_name,
        "total_rows": total_rows,
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "success_rate": success_rate,
    }
    _save_report(report, settings, report_name)
    print(f"[quality] {report_name}: {passed}/{len(checks)} checks passed (rate={success_rate:.0%})")
    return report


def _save_report(report: dict[str, Any], settings: Settings, report_name: str) -> None:
    out_path = settings.paths.quality_dir / f"{report_name}.json"
    ensure_parent(out_path)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[quality] Report saved -> {out_path}")


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Build freshness report.

    Steps:
    1. Find latest and oldest published date.
    2. Count stale rows.
    3. Build payload: latest_published, oldest_published, stale_rows, total_rows, is_fresh.
    4. Write JSON report.
    """
    total_rows = len(df)
    threshold = settings.freshness_threshold_days

    if total_rows == 0:
        payload: dict[str, Any] = {
            "latest_published": None,
            "oldest_published": None,
            "stale_rows": 0,
            "total_rows": 0,
            "stale_pct": 0.0,
            "freshness_threshold_days": threshold,
            "is_fresh": False,
        }
    else:
        published_series = df["published"].astype(str).str.strip()
        published_valid = published_series[published_series != ""]

        latest_published = published_valid.max() if not published_valid.empty else None
        oldest_published = published_valid.min() if not published_valid.empty else None

        stale_rows = int((df["age_days"] > threshold).sum())
        stale_pct = round(stale_rows / total_rows * 100, 1)
        is_fresh = stale_rows == 0

        payload = {
            "latest_published": latest_published,
            "oldest_published": oldest_published,
            "stale_rows": stale_rows,
            "total_rows": total_rows,
            "stale_pct": stale_pct,
            "freshness_threshold_days": threshold,
            "is_fresh": is_fresh,
        }

    ensure_parent(Path(report_path))
    Path(report_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    status = "FRESH" if payload.get("is_fresh") else "STALE"
    print(f"[freshness] Status={status}, stale={payload['stale_rows']}/{total_rows} -> {report_path}")
    return payload
