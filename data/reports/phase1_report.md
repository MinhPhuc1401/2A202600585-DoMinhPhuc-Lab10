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
| completeness | ✅ PASS | Total rows: 23. Missing paper_id: 0, title: 0, summary: 0 |
| accuracy | ✅ PASS | Published dates in the future: 0 (business rule: published <= today) |
| consistency | ✅ PASS | Dates not in YYYY-MM-DD format: 0 |
| timeliness | ✅ PASS | Stale rows (age > 180 days): 0 (0.0%) |
| validity | ✅ PASS | Rows with summary < 20 chars: 0 (domain rule: summary >= 20 chars) |
| uniqueness | ✅ PASS | Duplicate paper_id: 0 (dedup rate: 100.0%) |

## 4. Freshness Report

**Status**: ✅ FRESH

| Field | Value |
|-------|-------|
| Latest Published | 2026-06-02 |
| Oldest Published | 2025-12-19 |
| Total Rows | 23 |
| Stale Rows | 0 (0.0%) |
| Freshness Threshold | 180 days |
