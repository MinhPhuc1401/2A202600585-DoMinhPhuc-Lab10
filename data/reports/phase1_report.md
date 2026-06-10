# Phase 1 – Baseline Pipeline Report

## 1. Data Source

- **Source API**: Crossref REST API
- **Query**: `agentic retrieval augmented generation large language model`
- **Filter**: `from-pub-date:2025-12-12,has-abstract:true`
- **Raw records fetched**: 23
- **Clean records**: 23

## 2. Evaluation Metrics

| Metric | Value |
|--------|-------|
| Samples | 32 |
| Retrieval Hit Rate | 1.0000 |
| Mean Token F1 | 0.7500 |
| Judge Accuracy | 0.7500 |
| Mean Judge Score | 4 |

> **RAGAS**: Set RUN_RAGAS=1 to enable the slower Ragas pass.

## 3. Data Quality Checks

**Result**: 6/6 checks passed (success rate: 100%)

| Check | Status | Detail |
|-------|--------|--------|
| row_count_gt_zero | ✅ PASS | Total rows: 23 |
| paper_id_not_null | ✅ PASS | Null/empty paper_id count: 0 |
| paper_id_unique | ✅ PASS | Duplicate paper_id count: 0 |
| title_not_null | ✅ PASS | Null/empty title count: 0 |
| summary_min_length_20 | ✅ PASS | Rows with summary < 20 chars: 0 |
| freshness_age_lte_180_days | ✅ PASS | Stale rows (age > 180 days): 0 (0.0%) |

## 4. Freshness Report

**Status**: ✅ FRESH

| Field | Value |
|-------|-------|
| Latest Published | 2026-06-02 |
| Oldest Published | 2025-12-19 |
| Total Rows | 23 |
| Stale Rows | 0 (0.0%) |
| Freshness Threshold | 180 days |
