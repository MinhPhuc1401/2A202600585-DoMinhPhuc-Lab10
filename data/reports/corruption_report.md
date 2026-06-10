# Corruption & Repair – Comparison Report

## 1. Evaluation Metrics Comparison

| Metric | Baseline | Corrupted | Repaired | Delta (Repair vs Corrupted) |
|--------|----------|-----------|----------|-----------------------------|
| Retrieval Hit Rate | 1.0000 | 0.7500 | 1.0000 | +0.2500 |
| Mean Token F1 | 0.7500 | 0.5247 | 0.7500 | +0.2253 |
| Judge Accuracy | 0.7500 | 0.5312 | 0.7500 | +0.2188 |
| Mean Judge Score | 4.00 | 3.06 | 4.00 | +0.94 |

## 2. Data Quality Comparison

| Dataset | Checks Passed | Success Rate |
|---------|---------------|--------------|
| Corrupted | 3/6 | 50% |
| Repaired | 6/6 | 100% |

## 3. Freshness Comparison

| Dataset | Status | Latest | Oldest | Stale Rows |
|---------|--------|--------|--------|------------|
| Corrupted | FRESH | 2026-05-06 | 2020-01-01 | 0 (0.0%) |
| Repaired | FRESH | 2026-06-02 | 2025-12-19 | 0 (0.0%) |

## 4. Analysis

### Impact of Corruption

- Retrieval Hit Rate dropped by **0.2500** after corruption (1.0000 -> 0.7500), demonstrating that data quality directly affects RAG retrieval performance.
- After repair, Retrieval Hit Rate recovered by **0.2500** (0.7500 -> 1.0000), confirming that restoring clean data restores agent performance.
- Repaired dataset is **FRESH**, matching baseline data quality.
