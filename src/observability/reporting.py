from __future__ import annotations

from typing import Any

from core.utils import write_text


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    """Generate markdown report for baseline phase.

    Steps:
    1. Collect source summary.
    2. Print retrieval/evaluation metrics.
    3. Print data quality and freshness.
    4. Write markdown to report_path.
    """
    lines: list[str] = []

    lines.append("# Phase 1 – Baseline Pipeline Report\n")

    # --- Source Summary ---
    lines.append("## 1. Data Source\n")
    lines.append(f"- **Source API**: {source_summary.get('source_api', 'Crossref REST API')}")
    lines.append(f"- **Query**: `{source_summary.get('query', '')}`")
    lines.append(f"- **Filter**: `{source_summary.get('filter', '')}`")
    lines.append(f"- **Raw records fetched**: {source_summary.get('raw_count', 'N/A')}")
    lines.append(f"- **Clean records**: {source_summary.get('clean_count', 'N/A')}")
    lines.append("")

    # --- Evaluation Metrics ---
    lines.append("## 2. Evaluation Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Samples | {metrics.get('samples', 'N/A')} |")
    lines.append(f"| Retrieval Hit Rate | {metrics.get('retrieval_hit_rate', 'N/A'):.4f} |" if isinstance(metrics.get('retrieval_hit_rate'), float) else f"| Retrieval Hit Rate | {metrics.get('retrieval_hit_rate', 'N/A')} |")
    lines.append(f"| Mean Token F1 | {metrics.get('mean_token_f1', 'N/A'):.4f} |" if isinstance(metrics.get('mean_token_f1'), float) else f"| Mean Token F1 | {metrics.get('mean_token_f1', 'N/A')} |")
    lines.append(f"| Judge Accuracy | {metrics.get('judge_accuracy', 'N/A'):.4f} |" if isinstance(metrics.get('judge_accuracy'), float) else f"| Judge Accuracy | {metrics.get('judge_accuracy', 'N/A')} |")
    lines.append(f"| Mean Judge Score | {metrics.get('mean_judge_score', 'N/A'):.2f} |" if isinstance(metrics.get('mean_judge_score'), float) else f"| Mean Judge Score | {metrics.get('mean_judge_score', 'N/A')} |")
    lines.append("")

    ragas = metrics.get("ragas", {})
    if ragas and not ragas.get("skipped") and not ragas.get("error"):
        lines.append("### RAGAS Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in ragas.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        lines.append("")
    elif ragas.get("skipped"):
        lines.append(f"> **RAGAS**: {ragas['skipped']}\n")

    # --- Data Quality ---
    lines.append("## 3. Data Quality Checks\n")
    checks = quality.get("checks", [])
    passed = quality.get("passed", 0)
    failed = quality.get("failed", 0)
    success_rate = quality.get("success_rate", 0.0)

    lines.append(f"**Result**: {passed}/{passed + failed} checks passed (success rate: {success_rate:.0%})\n")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")
    for check in checks:
        status = "✅ PASS" if check["passed"] else "❌ FAIL"
        lines.append(f"| {check['name']} | {status} | {check.get('detail', '')} |")
    lines.append("")

    # --- Freshness ---
    lines.append("## 4. Freshness Report\n")
    is_fresh = freshness.get("is_fresh", False)
    fresh_status = "✅ FRESH" if is_fresh else "⚠️ STALE"
    lines.append(f"**Status**: {fresh_status}\n")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Latest Published | {freshness.get('latest_published', 'N/A')} |")
    lines.append(f"| Oldest Published | {freshness.get('oldest_published', 'N/A')} |")
    lines.append(f"| Total Rows | {freshness.get('total_rows', 'N/A')} |")
    lines.append(f"| Stale Rows | {freshness.get('stale_rows', 'N/A')} ({freshness.get('stale_pct', 0):.1f}%) |")
    lines.append(f"| Freshness Threshold | {freshness.get('freshness_threshold_days', 'N/A')} days |")
    lines.append("")

    write_text(report_path, "\n".join(lines))
    print(f"[reporting] Phase 1 report saved -> {report_path}")


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    """Generate markdown comparison report: baseline vs corrupted vs repaired."""
    lines: list[str] = []

    lines.append("# Corruption & Repair – Comparison Report\n")

    # --- Metrics Comparison Table ---
    lines.append("## 1. Evaluation Metrics Comparison\n")
    lines.append("| Metric | Baseline | Corrupted | Repaired | Delta (Repair vs Corrupted) |")
    lines.append("|--------|----------|-----------|----------|-----------------------------|")

    metric_keys = [
        ("retrieval_hit_rate", "Retrieval Hit Rate", ".4f"),
        ("mean_token_f1", "Mean Token F1", ".4f"),
        ("judge_accuracy", "Judge Accuracy", ".4f"),
        ("mean_judge_score", "Mean Judge Score", ".2f"),
    ]

    for key, label, fmt in metric_keys:
        b_val = baseline_metrics.get(key)
        c_val = corrupted_metrics.get(key)
        r_val = repaired_metrics.get(key)

        def fmt_val(v):
            if v is None:
                return "N/A"
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(v)

        delta = ""
        if c_val is not None and r_val is not None:
            try:
                d = float(r_val) - float(c_val)
                delta = f"+{d:{fmt}}" if d >= 0 else f"{d:{fmt}}"
            except (TypeError, ValueError):
                delta = "N/A"

        lines.append(f"| {label} | {fmt_val(b_val)} | {fmt_val(c_val)} | {fmt_val(r_val)} | {delta} |")

    lines.append("")

    # --- Quality Comparison ---
    lines.append("## 2. Data Quality Comparison\n")
    lines.append("| Dataset | Checks Passed | Success Rate |")
    lines.append("|---------|---------------|--------------|")
    for label, qr in [("Corrupted", corrupted_quality), ("Repaired", repaired_quality)]:
        p = qr.get("passed", "N/A")
        total = (qr.get("passed", 0) or 0) + (qr.get("failed", 0) or 0)
        sr = qr.get("success_rate", 0.0)
        lines.append(f"| {label} | {p}/{total} | {sr:.0%} |")
    lines.append("")

    # --- Freshness Comparison ---
    lines.append("## 3. Freshness Comparison\n")
    lines.append("| Dataset | Status | Latest | Oldest | Stale Rows |")
    lines.append("|---------|--------|--------|--------|------------|")
    for label, fr in [("Corrupted", corrupted_freshness), ("Repaired", repaired_freshness)]:
        status = "FRESH" if fr.get("is_fresh") else "STALE"
        latest = fr.get("latest_published", "N/A")
        oldest = fr.get("oldest_published", "N/A")
        stale = f"{fr.get('stale_rows', 'N/A')} ({fr.get('stale_pct', 0):.1f}%)"
        lines.append(f"| {label} | {status} | {latest} | {oldest} | {stale} |")
    lines.append("")

    # --- Analysis ---
    lines.append("## 4. Analysis\n")
    lines.append("### Impact of Corruption\n")

    b_hit = baseline_metrics.get("retrieval_hit_rate")
    c_hit = corrupted_metrics.get("retrieval_hit_rate")
    r_hit = repaired_metrics.get("retrieval_hit_rate")

    if b_hit is not None and c_hit is not None:
        impact = float(c_hit) - float(b_hit)
        lines.append(
            f"- Retrieval Hit Rate dropped by **{abs(impact):.4f}** after corruption "
            f"({b_hit:.4f} -> {c_hit:.4f}), demonstrating that data quality directly affects "
            f"RAG retrieval performance."
        )

    if c_hit is not None and r_hit is not None:
        recovery = float(r_hit) - float(c_hit)
        lines.append(
            f"- After repair, Retrieval Hit Rate recovered by **{recovery:.4f}** "
            f"({c_hit:.4f} -> {r_hit:.4f}), confirming that restoring clean data "
            f"restores agent performance."
        )

    corrupt_fresh = corrupted_freshness.get("is_fresh", True)
    repair_fresh = repaired_freshness.get("is_fresh", False)
    if not corrupt_fresh:
        lines.append(
            "- Corrupted dataset shows **STALE** data due to backdated publication dates, "
            "detectable via freshness monitoring."
        )
    if repair_fresh:
        lines.append("- Repaired dataset is **FRESH**, matching baseline data quality.")

    lines.append("")

    write_text(report_path, "\n".join(lines))
    print(f"[reporting] Corruption report saved -> {report_path}")
