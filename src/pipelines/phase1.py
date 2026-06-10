from __future__ import annotations

from datetime import datetime, UTC

from core.config import load_settings
from core.utils import write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import fetch_source_records, load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_phase1_report
from retrieval.index import LocalEmbeddingIndex


def main() -> None:
    """Baseline pipeline end-to-end.

    Steps:
    1. Load settings.
    2. Load or fetch raw records.
    3. Clean data.
    4. Save clean CSV/JSON.
    5. Build Chroma index.
    6. Create or load evaluation set.
    7. Evaluate.
    8. Run quality checks and freshness report.
    9. Generate markdown report.
    10. Demo agent on sample questions.
    """
    print("=" * 60)
    print("PHASE 1 – BASELINE PIPELINE")
    print("=" * 60)

    settings = load_settings()
    run_date = datetime.now(UTC)

    # --- Step 2: Load or fetch raw records ---
    raw_path = settings.paths.raw_records_json
    if settings.refresh_source or not raw_path.exists():
        print("\n[phase1] Fetching raw records from source...")
        records = fetch_source_records(settings)
    else:
        print(f"\n[phase1] Loading cached raw records from {raw_path}")
        records = load_raw_records(raw_path)
    print(f"[phase1] Raw records: {len(records)}")

    # --- Step 3: Clean data ---
    print("\n[phase1] Cleaning data...")
    df = build_clean_dataframe(records, run_date)
    print(f"[phase1] Cleaned: {len(df)} rows")

    # --- Step 4: Save clean CSV/JSON ---
    write_csv(df, settings.paths.clean_csv)
    write_json(settings.paths.clean_json, df.to_dict(orient="records"))
    print(f"[phase1] Saved clean data -> {settings.paths.clean_csv}")

    # --- Step 5: Build Chroma index ---
    print("\n[phase1] Building embedding index...")
    index = LocalEmbeddingIndex.build(
        df=df,
        settings=settings,
        embeddings_output_path=settings.paths.embeddings_json,
    )
    print(f"[phase1] Index built: {len(index.documents)} documents")

    # --- Step 6: Create or load evaluation set ---
    testset_path = settings.paths.eval_testset
    if settings.refresh_test_set or not testset_path.exists():
        print("\n[phase1] Building evaluation test set...")
        build_test_set(df, testset_path)
    else:
        print(f"\n[phase1] Using cached test set: {testset_path}")

    # --- Step 7: Evaluate ---
    print("\n[phase1] Running evaluation...")
    bundle = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=testset_path,
        metrics_output_path=settings.paths.baseline_metrics,
        answers_output_path=settings.paths.baseline_answers,
    )
    metrics = bundle.summary
    print(f"[phase1] Evaluation done:")
    print(f"  retrieval_hit_rate : {metrics.get('retrieval_hit_rate', 'N/A'):.4f}")
    print(f"  mean_token_f1      : {metrics.get('mean_token_f1', 'N/A'):.4f}")
    print(f"  judge_accuracy     : {metrics.get('judge_accuracy', 'N/A'):.4f}")
    print(f"  mean_judge_score   : {metrics.get('mean_judge_score', 'N/A'):.2f}")

    # --- Step 8: Quality checks + freshness ---
    print("\n[phase1] Running data quality checks...")
    quality = run_data_quality_checks(df, settings, report_name="baseline")

    print("\n[phase1] Building freshness report...")
    freshness = build_freshness_report(df, settings, settings.paths.freshness_report)

    # --- Step 9: Generate markdown report ---
    print("\n[phase1] Generating baseline report...")
    source_summary = {
        "source_api": settings.source_api,
        "query": settings.source_query,
        "filter": settings.source_filter,
        "raw_count": len(records),
        "clean_count": len(df),
    }
    generate_phase1_report(
        report_path=settings.paths.baseline_report,
        source_summary=source_summary,
        metrics=metrics,
        quality=quality,
        freshness=freshness,
    )

    # --- Step 10: Demo agent on sample questions ---
    print("\n[phase1] Demo agent answers...")
    from retrieval.qa import answer_question

    demo_questions = [
        f"What is the paper '{df['title'].iloc[0]}' about?",
        f"Who authored the paper '{df['title'].iloc[1]}'?",
        "What are the latest trends in retrieval augmented generation?",
    ]
    demo_answers = []
    for q in demo_questions:
        result = answer_question(q, settings=settings, index=index)
        print(f"  Q: {q[:70]}")
        print(f"  A: {result.answer[:120]}")
        print()
        demo_answers.append({"question": q, "answer": result.answer})

    write_json(settings.paths.demo_answers, demo_answers)

    print("=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"  Reports -> {settings.paths.baseline_report}")
    print(f"  Metrics -> {settings.paths.baseline_metrics}")
    print("=" * 60)
