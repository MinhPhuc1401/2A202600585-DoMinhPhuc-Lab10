from __future__ import annotations

from datetime import datetime, UTC

import pandas as pd

from core.config import load_settings
from core.utils import read_json, write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from ingestion.cleaning import build_clean_dataframe
from ingestion.corruption import corrupt_clean_dataframe
from ingestion.crossref import load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_corruption_report
from retrieval.index import LocalEmbeddingIndex


def main() -> None:
    """Corruption -> evaluate -> repair -> compare flow.

    Steps:
    1. Load baseline metrics and clean dataset.
    2. Create corrupted dataframe.
    3. Save corrupted artifacts.
    4. Rebuild index and evaluate.
    5. Run quality checks/freshness on corrupted data.
    6. Repair from raw records.
    7. Evaluate repaired dataset.
    8. Generate comparison report.
    """
    print("=" * 60)
    print("CORRUPTION FLOW PIPELINE")
    print("=" * 60)

    settings = load_settings()
    run_date = datetime.now(UTC)

    # --- Step 1: Load baseline metrics and clean dataset ---
    print("\n[corruption_flow] Loading baseline artifacts...")

    if not settings.paths.baseline_metrics.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found: {settings.paths.baseline_metrics}\n"
            "Run phase1 first: python script/run_phase1.py"
        )
    baseline_metrics = read_json(settings.paths.baseline_metrics)
    print(f"[corruption_flow] Baseline metrics loaded (retrieval_hit_rate={baseline_metrics.get('retrieval_hit_rate', 'N/A')})")

    if not settings.paths.clean_csv.exists():
        raise FileNotFoundError(f"Clean dataset not found: {settings.paths.clean_csv}")
    df_clean = pd.read_csv(settings.paths.clean_csv)
    # Restore list columns from string representation
    import ast
    for col in ["authors", "categories"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
            )
    print(f"[corruption_flow] Clean dataset loaded: {len(df_clean)} rows")

    # --- Step 2: Create corrupted dataframe ---
    print("\n[corruption_flow] Corrupting data...")
    df_corrupted = corrupt_clean_dataframe(df_clean, settings.paths.corruption_log)
    print(f"[corruption_flow] Corrupted dataset: {len(df_corrupted)} rows")

    # --- Step 3: Save corrupted artifacts ---
    write_csv(df_corrupted, settings.paths.corrupted_clean_csv)
    write_json(settings.paths.corrupted_clean_json, df_corrupted.to_dict(orient="records"))
    print(f"[corruption_flow] Saved corrupted data -> {settings.paths.corrupted_clean_csv}")

    # --- Step 4: Rebuild index on corrupted data and evaluate ---
    print("\n[corruption_flow] Building corrupted embedding index...")
    corrupted_index = LocalEmbeddingIndex.build(
        df=df_corrupted,
        settings=settings,
        embeddings_output_path=settings.paths.corrupted_embeddings_json,
    )
    print(f"[corruption_flow] Corrupted index: {len(corrupted_index.documents)} documents")

    print("\n[corruption_flow] Evaluating corrupted pipeline...")
    corrupted_bundle = evaluate_pipeline(
        settings=settings,
        index=corrupted_index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.corrupted_metrics,
        answers_output_path=settings.paths.corrupted_answers,
    )
    corrupted_metrics = corrupted_bundle.summary
    print(f"[corruption_flow] Corrupted evaluation:")
    print(f"  retrieval_hit_rate : {corrupted_metrics.get('retrieval_hit_rate', 'N/A'):.4f}")
    print(f"  judge_accuracy     : {corrupted_metrics.get('judge_accuracy', 'N/A'):.4f}")

    # --- Step 5: Quality checks + freshness on corrupted data ---
    print("\n[corruption_flow] Running quality checks on corrupted data...")
    corrupted_quality = run_data_quality_checks(df_corrupted, settings, report_name="corrupted")
    corrupted_freshness = build_freshness_report(
        df_corrupted,
        settings,
        settings.paths.quality_dir / "freshness_corrupted.json",
    )

    # --- Step 6: Repair from raw source ---
    print("\n[corruption_flow] Repairing corrupted data from raw source...")
    if not settings.paths.raw_records_json.exists():
        raise FileNotFoundError(
            f"Raw records not found: {settings.paths.raw_records_json}\n"
            "Run phase1 first to fetch raw data."
        )
    raw_records = load_raw_records(settings.paths.raw_records_json)
    df_repaired = build_clean_dataframe(raw_records, run_date)
    print(f"[corruption_flow] Repaired dataset: {len(df_repaired)} rows (vs baseline {len(df_clean)})") 

    write_csv(df_repaired, settings.paths.repaired_clean_csv)
    write_json(settings.paths.repaired_clean_json, df_repaired.to_dict(orient="records"))
    print(f"[corruption_flow] Saved repaired data -> {settings.paths.repaired_clean_csv}")

    # --- Step 7: Evaluate repaired dataset ---
    print("\n[corruption_flow] Building repaired embedding index...")
    repaired_index = LocalEmbeddingIndex.build(
        df=df_repaired,
        settings=settings,
        embeddings_output_path=settings.paths.repaired_embeddings_json,
    )

    print("\n[corruption_flow] Evaluating repaired pipeline...")
    repaired_bundle = evaluate_pipeline(
        settings=settings,
        index=repaired_index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.repaired_metrics,
        answers_output_path=settings.paths.repaired_answers,
    )
    repaired_metrics = repaired_bundle.summary
    print(f"[corruption_flow] Repaired evaluation:")
    print(f"  retrieval_hit_rate : {repaired_metrics.get('retrieval_hit_rate', 'N/A'):.4f}")
    print(f"  judge_accuracy     : {repaired_metrics.get('judge_accuracy', 'N/A'):.4f}")

    # --- Step 7b: Quality checks + freshness on repaired data ---
    print("\n[corruption_flow] Running quality checks on repaired data...")
    repaired_quality = run_data_quality_checks(df_repaired, settings, report_name="repaired")
    repaired_freshness = build_freshness_report(
        df_repaired,
        settings,
        settings.paths.quality_dir / "freshness_repaired.json",
    )

    # --- Step 8: Generate comparison report ---
    print("\n[corruption_flow] Generating comparison report...")
    generate_corruption_report(
        report_path=settings.paths.comparison_report,
        baseline_metrics=baseline_metrics,
        corrupted_metrics=corrupted_metrics,
        repaired_metrics=repaired_metrics,
        corrupted_quality=corrupted_quality,
        repaired_quality=repaired_quality,
        corrupted_freshness=corrupted_freshness,
        repaired_freshness=repaired_freshness,
    )

    # Print summary comparison
    print("\n[corruption_flow] === METRICS COMPARISON ===")
    print(f"{'Metric':<25} {'Baseline':>10} {'Corrupted':>10} {'Repaired':>10}")
    print("-" * 60)
    for key in ["retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"]:
        b = baseline_metrics.get(key, 0)
        c = corrupted_metrics.get(key, 0)
        r = repaired_metrics.get(key, 0)
        try:
            print(f"{key:<25} {float(b):>10.4f} {float(c):>10.4f} {float(r):>10.4f}")
        except (TypeError, ValueError):
            print(f"{key:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print("\n" + "=" * 60)
    print("CORRUPTION FLOW COMPLETE")
    print(f"  Comparison report -> {settings.paths.comparison_report}")
    print("=" * 60)
