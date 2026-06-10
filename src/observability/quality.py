from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings
from core.utils import ensure_parent


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Run data quality checks based on the 6 standard data quality dimensions.

    Dimensions:
    1. Completeness  – No missing records or critical fields.
    2. Accuracy      – Data matches reality / source of truth.
    3. Consistency   – Same entity has the same format across the dataset.
    4. Timeliness    – Data is fresh enough for the use case.
    5. Validity      – Data follows expected format and domain rules.
    6. Uniqueness    – No duplicate records.
    """
    import re
    from datetime import datetime

    checks: list[dict[str, Any]] = []
    total_rows = len(df)

    # ── 1. COMPLETENESS ──────────────────────────────────────────────────────
    # "Không thiếu records hoặc fields quan trọng"
    # Check: row count > 0, critical fields (paper_id, title, summary) not null
    if total_rows == 0:
        checks.append({
            "name": "completeness",
            "dimension": "Completeness",
            "passed": False,
            "detail": "Dataset is empty (0 rows)",
        })
        report = {
            "report_name": report_name,
            "total_rows": total_rows,
            "checks": checks,
            "passed": 0,
            "failed": 1,
            "success_rate": 0.0,
        }
        _save_report(report, settings, report_name)
        return report

    null_pid = int(df["paper_id"].isna().sum() + (df["paper_id"].astype(str).str.strip() == "").sum())
    null_title = int(df["title"].isna().sum() + (df["title"].astype(str).str.strip() == "").sum())
    null_summary = int((df["summary"].astype(str).str.strip() == "").sum())
    total_missing = null_pid + null_title + null_summary
    completeness_ok = total_missing == 0
    checks.append({
        "name": "completeness",
        "dimension": "Completeness",
        "passed": completeness_ok,
        "detail": (
            f"Total rows: {total_rows}. "
            f"Missing paper_id: {null_pid}, title: {null_title}, summary: {null_summary}"
        ),
    })

    # ── 2. ACCURACY ──────────────────────────────────────────────────────────
    # "Data đúng với thực tế. Check: validate với nguồn gốc, business rules"
    # Check: published date must not be in the future (business rule)
    today_str = datetime.now().strftime("%Y-%m-%d")
    future_dates = 0
    if "published" in df.columns:
        pub_str = df["published"].astype(str).str.strip()
        valid_dates = pub_str[pub_str.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
        future_dates = int((valid_dates > today_str).sum())
    accuracy_ok = future_dates == 0
    checks.append({
        "name": "accuracy",
        "dimension": "Accuracy",
        "passed": accuracy_ok,
        "detail": f"Published dates in the future: {future_dates} (business rule: published <= today)",
    })

    # ── 3. CONSISTENCY ───────────────────────────────────────────────────────
    # "Cùng entity, cùng format across systems. Check: cross-system reconciliation"
    # Check: all published dates follow YYYY-MM-DD format consistently
    inconsistent_dates = 0
    if "published" in df.columns:
        pub_str = df["published"].astype(str).str.strip()
        non_empty = pub_str[pub_str != ""]
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        inconsistent_dates = int((~non_empty.str.match(date_pattern, na=False)).sum())
    consistency_ok = inconsistent_dates == 0
    checks.append({
        "name": "consistency",
        "dimension": "Consistency",
        "passed": consistency_ok,
        "detail": f"Dates not in YYYY-MM-DD format: {inconsistent_dates}",
    })

    # ── 4. TIMELINESS ────────────────────────────────────────────────────────
    # "Data đủ fresh cho use case. Check: max age, last-updated timestamp"
    # Check: age_days <= freshness_threshold_days
    threshold = settings.freshness_threshold_days
    stale_count = int((df["age_days"] > threshold).sum())
    stale_pct = round(stale_count / total_rows * 100, 1) if total_rows > 0 else 0.0
    timeliness_ok = stale_count == 0
    checks.append({
        "name": "timeliness",
        "dimension": "Timeliness",
        "passed": timeliness_ok,
        "detail": f"Stale rows (age > {threshold} days): {stale_count} ({stale_pct}%)",
    })

    # ── 5. VALIDITY ──────────────────────────────────────────────────────────
    # "Data theo đúng format và domain rules. Check: regex patterns, range checks"
    # Check: summary must have >= 20 chars (domain rule for meaningful content)
    short_summary = int((df["summary_chars"] < 20).sum())
    validity_ok = short_summary == 0
    checks.append({
        "name": "validity",
        "dimension": "Validity",
        "passed": validity_ok,
        "detail": f"Rows with summary < 20 chars: {short_summary} (domain rule: summary >= 20 chars)",
    })

    # ── 6. UNIQUENESS ────────────────────────────────────────────────────────
    # "Không có duplicates. Check: dedup rate, composite key uniqueness"
    # Check: paper_id must be unique (primary key)
    dup_count = int(df["paper_id"].duplicated().sum())
    dedup_rate = round((1 - dup_count / total_rows) * 100, 1) if total_rows > 0 else 100.0
    uniqueness_ok = dup_count == 0
    checks.append({
        "name": "uniqueness",
        "dimension": "Uniqueness",
        "passed": uniqueness_ok,
        "detail": f"Duplicate paper_id: {dup_count} (dedup rate: {dedup_rate}%)",
    })

    # ── Summary ──────────────────────────────────────────────────────────────
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
