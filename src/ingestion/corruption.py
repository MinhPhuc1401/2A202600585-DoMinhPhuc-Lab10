from __future__ import annotations

import json
import random
import string
from pathlib import Path

import pandas as pd

from core.utils import ensure_parent


def _inject_noise(text: str, noise_ratio: float = 0.05) -> str:
    """Inject random printable ASCII characters into a string."""
    if not text:
        return text
    chars = list(text)
    n_noise = max(1, int(len(chars) * noise_ratio))
    positions = random.sample(range(len(chars)), min(n_noise, len(chars)))
    for pos in positions:
        chars[pos] = random.choice(string.ascii_letters + string.digits + "!@#$%^&*")
    return "".join(chars)


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path) -> pd.DataFrame:
    """Simulate multiple types of data corruption.

    Steps:
    1. Drop ~20% of latest records (newest rows).
    2. Blank summary on ~15% of rows.
    3. Inject noise into summary text.
    4. Truncate title to first 5 words on ~10% of rows.
    5. Stale published date: set year to 2020 on ~10% of rows.
    6. Add 2-3 duplicate rows.
    7. Rebuild text_for_embedding.
    8. Write corruption log to output_log_path.
    """
    random.seed(42)
    corrupted = df.copy()
    log: dict = {
        "original_rows": len(df),
        "corruptions": [],
    }

    total = len(corrupted)

    # --- 1. Drop ~20% of latest records (sorted newest-first, so drop from top) ---
    n_drop = max(1, int(total * 0.20))
    dropped_ids = corrupted.head(n_drop)["paper_id"].tolist()
    corrupted = corrupted.iloc[n_drop:].reset_index(drop=True)
    log["corruptions"].append({
        "type": "drop_latest_records",
        "count": n_drop,
        "dropped_paper_ids": dropped_ids,
    })
    print(f"[corruption] Dropped {n_drop} latest records")

    # --- 2. Blank summary on ~15% of remaining rows ---
    n_blank = max(1, int(len(corrupted) * 0.15))
    blank_idx = random.sample(range(len(corrupted)), min(n_blank, len(corrupted)))
    corrupted.loc[blank_idx, "summary"] = ""
    corrupted.loc[blank_idx, "summary_chars"] = 0
    log["corruptions"].append({
        "type": "blank_summary",
        "count": n_blank,
        "row_indices": blank_idx,
    })
    print(f"[corruption] Blanked summary on {n_blank} rows")

    # --- 3. Inject noise into ~20% of non-blank summaries ---
    non_blank_mask = corrupted["summary"].str.len() > 0
    non_blank_idx = corrupted[non_blank_mask].index.tolist()
    n_noise = max(1, int(len(non_blank_idx) * 0.20))
    noise_idx = random.sample(non_blank_idx, min(n_noise, len(non_blank_idx)))
    corrupted.loc[noise_idx, "summary"] = corrupted.loc[noise_idx, "summary"].apply(_inject_noise)
    log["corruptions"].append({
        "type": "inject_noise_summary",
        "count": n_noise,
        "row_indices": list(noise_idx),
    })
    print(f"[corruption] Injected noise into {n_noise} summaries")

    # --- 4. Truncate title to first 5 words on ~10% of rows ---
    n_trunc = max(1, int(len(corrupted) * 0.10))
    trunc_idx = random.sample(range(len(corrupted)), min(n_trunc, len(corrupted)))
    def truncate_title(t: str) -> str:
        words = t.split()
        return " ".join(words[:5]) + ("..." if len(words) > 5 else "")
    corrupted.loc[trunc_idx, "title"] = corrupted.loc[trunc_idx, "title"].apply(truncate_title)
    log["corruptions"].append({
        "type": "truncate_title",
        "count": n_trunc,
        "row_indices": trunc_idx,
    })
    print(f"[corruption] Truncated title on {n_trunc} rows")

    # --- 5. Stale published date (set year to 2020) on ~10% of rows ---
    n_stale = max(1, int(len(corrupted) * 0.10))
    stale_idx = random.sample(range(len(corrupted)), min(n_stale, len(corrupted)))
    def stale_date(d: str) -> str:
        if d and len(d) >= 4:
            return "2020" + d[4:]
        return "2020-01-01"
    corrupted.loc[stale_idx, "published"] = corrupted.loc[stale_idx, "published"].apply(stale_date)
    log["corruptions"].append({
        "type": "stale_published_date",
        "count": n_stale,
        "row_indices": stale_idx,
    })
    print(f"[corruption] Staled published date on {n_stale} rows")

    # --- 6. Add 2 duplicate rows ---
    n_dup = min(2, len(corrupted))
    dup_rows = corrupted.sample(n=n_dup, random_state=42)
    corrupted = pd.concat([corrupted, dup_rows], ignore_index=True)
    log["corruptions"].append({
        "type": "duplicate_rows",
        "count": n_dup,
        "duplicated_paper_ids": dup_rows["paper_id"].tolist(),
    })
    print(f"[corruption] Added {n_dup} duplicate rows")

    # --- 7. Rebuild text_for_embedding ---
    # Convert all text columns to str and fill NaN to avoid float64 concat errors
    for col in ["title", "summary", "authors_joined", "categories_joined"]:
        corrupted[col] = corrupted[col].fillna("").astype(str)

    corrupted["summary_chars"] = corrupted["summary"].str.len().fillna(0).astype(int)
    corrupted["text_for_embedding"] = (
        "Title: " + corrupted["title"] + "\n"
        + "Authors: " + corrupted["authors_joined"] + "\n"
        + "Categories: " + corrupted["categories_joined"] + "\n"
        + "Summary: " + corrupted["summary"]
    )

    # --- 8. Write corruption log ---
    log["corrupted_rows"] = len(corrupted)
    ensure_parent(Path(output_log_path))
    Path(output_log_path).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[corruption] Corruption log saved -> {output_log_path}")

    return corrupted


def partial_repair_dataframe(
    df_corrupted: pd.DataFrame,
    raw_records_lookup: dict[str, dict],
) -> pd.DataFrame:
    """Partially repair corrupted dataframe — intentionally imperfect.

    FIXED (automated, easy repairs):
    - Blank summaries  -> restore from raw
    - Duplicate rows   -> drop_duplicates
    - Stale dates      -> restore from raw

    NOT FIXED (residual damage):
    - Dropped records  -> NOT re-added (permanent data loss)
    - Truncated titles -> NOT restored (assumed undetected)
    - Noisy summaries  -> NOT cleaned (subtle noise, hard to detect)

    Result: baseline > repaired > corrupted
    """
    repaired = df_corrupted.copy()

    # Ensure text columns are strings
    for col in ["title", "summary", "authors_joined", "categories_joined", "published"]:
        repaired[col] = repaired[col].fillna("").astype(str)

    # --- Fix 1: Remove duplicates ---
    before = len(repaired)
    repaired = repaired.drop_duplicates(subset=["paper_id"], keep="first").reset_index(drop=True)
    n_dedup = before - len(repaired)
    if n_dedup > 0:
        print(f"[repair] Removed {n_dedup} duplicate rows")

    # --- Fix 2: Restore blank summaries ---
    blank_mask = repaired["summary"].str.strip() == ""
    n_blank = int(blank_mask.sum())
    restored = 0
    for idx in repaired[blank_mask].index:
        pid = str(repaired.at[idx, "paper_id"])
        raw = raw_records_lookup.get(pid)
        if raw and raw.get("summary", "").strip():
            repaired.at[idx, "summary"] = raw["summary"]
            restored += 1
    if n_blank > 0:
        print(f"[repair] Restored {restored}/{n_blank} blank summaries")

    # --- Fix 3: Restore stale published dates ---
    stale_mask = repaired["published"].str.startswith("2020")
    n_stale = int(stale_mask.sum())
    restored_dates = 0
    for idx in repaired[stale_mask].index:
        pid = str(repaired.at[idx, "paper_id"])
        raw = raw_records_lookup.get(pid)
        if raw and raw.get("published", "").strip() and not raw["published"].startswith("2020"):
            repaired.at[idx, "published"] = raw["published"]
            restored_dates += 1
    if n_stale > 0:
        print(f"[repair] Restored {restored_dates}/{n_stale} stale dates")

    # NOT fixed: dropped records, truncated titles, noisy summaries
    print(f"[repair] Residual damage kept: dropped records / truncated titles / noisy summaries")

    # Rebuild derived columns
    repaired["summary_chars"] = repaired["summary"].str.len().fillna(0).astype(int)
    repaired["text_for_embedding"] = (
        "Title: " + repaired["title"] + "\n"
        + "Authors: " + repaired["authors_joined"] + "\n"
        + "Categories: " + repaired["categories_joined"] + "\n"
        + "Summary: " + repaired["summary"]
    )

    print(f"[repair] Partial repair done: {len(repaired)} rows remaining")
    return repaired
