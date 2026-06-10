from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from core.utils import ensure_parent


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Build evaluation set from cleaned dataframe.

    Steps:
    1. Check minimum document count.
    2. Select representative papers (up to 8, diverse).
    3. Create multiple question types per paper:
       - summary   : "What is the paper '...' about?"
       - authors   : "Who authored the paper '...'?"
       - date      : "When was the paper '...' published?"
       - categories: "What categories does the paper '...' belong to?"
    4. Each row has: id, question_type, question, ground_truth, ground_truth_doc_ids.
    5. Write JSON to output_path.
    """
    MIN_DOCS = 4
    if len(df) < MIN_DOCS:
        raise ValueError(
            f"Not enough documents to build a test set: need at least {MIN_DOCS}, got {len(df)}."
        )

    # Select up to 8 representative papers
    # Sample from different positions to get diversity (head + tail + middle)
    n_sample = min(8, len(df))
    if len(df) <= n_sample:
        sample_df = df.copy()
    else:
        # Take from evenly spaced positions for diversity
        step = len(df) // n_sample
        indices = [i * step for i in range(n_sample)]
        sample_df = df.iloc[indices].copy()

    test_set: list[dict[str, Any]] = []

    for _, row in sample_df.iterrows():
        paper_id: str = str(row["paper_id"])
        title: str = str(row["title"])
        summary: str = str(row["summary"])
        authors_joined: str = str(row.get("authors_joined", ""))
        categories_joined: str = str(row.get("categories_joined", ""))
        published: str = str(row.get("published", ""))

        # Ground truth for each type
        # summary → first meaningful sentence of abstract
        summary_gt = summary.strip()
        if len(summary_gt) > 300:
            # Truncate to first sentence for a tighter ground truth
            import re
            sentences = re.split(r'(?<=[.!?])\s+', summary_gt)
            summary_gt = sentences[0] if sentences else summary_gt[:300]

        # authors → joined author list (or "Unknown" if empty)
        authors_gt = authors_joined if authors_joined.strip() else "Unknown"

        # date → published date string
        date_gt = published if published.strip() else "Unknown"

        # categories → joined categories (or "Not specified" if empty)
        categories_gt = categories_joined if categories_joined.strip() else "Not specified"

        # Build questions
        questions: list[tuple[str, str, str]] = [
            (
                "summary",
                f"What is the paper '{title}' about?",
                summary_gt,
            ),
            (
                "authors",
                f"Who authored the paper '{title}'?",
                authors_gt,
            ),
            (
                "date",
                f"When was the paper '{title}' published?",
                date_gt,
            ),
            (
                "categories",
                f"What categories does the paper '{title}' belong to?",
                categories_gt,
            ),
        ]

        for question_type, question, ground_truth in questions:
            test_set.append(
                {
                    "id": str(uuid.uuid4()),
                    "question_type": question_type,
                    "question": question,
                    "ground_truth": ground_truth,
                    "ground_truth_doc_ids": [paper_id],
                }
            )

    # Write JSON
    ensure_parent(Path(output_path))
    Path(output_path).write_text(
        json.dumps(test_set, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[testset] Built {len(test_set)} test samples from {n_sample} papers -> {output_path}")

    return test_set
