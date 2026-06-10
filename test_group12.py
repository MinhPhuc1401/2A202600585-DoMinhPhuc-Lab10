import sys
sys.path.insert(0, 'src')
from datetime import datetime, UTC
from core.config import load_settings
from ingestion.crossref import load_raw_records
from ingestion.cleaning import build_clean_dataframe
from evaluation.testset import build_test_set
from core.utils import write_csv, write_json

settings = load_settings()

# Load raw records (already fetched)
records = load_raw_records(settings.paths.raw_records_json)
print(f'Loaded {len(records)} raw records')

# Clean
df = build_clean_dataframe(records, datetime.now(UTC))
print(f'Cleaned: {len(df)} rows')
print(f'Columns: {list(df.columns)}')

age_min = df['age_days'].min()
age_max = df['age_days'].max()
print(f'Age days range: {age_min} - {age_max}')

sample_text = df['text_for_embedding'].iloc[0][:200]
print(f'Sample text_for_embedding:\n{sample_text}')

# Save
write_csv(df, settings.paths.clean_csv)
write_json(settings.paths.clean_json, df.to_dict(orient='records'))
print('Saved CSV + JSON')

# Build testset
test_set = build_test_set(df, settings.paths.eval_testset)
print(f'Test set: {len(test_set)} samples')
q0 = test_set[0]
print(f'Sample: {q0["question_type"]} -> {q0["question"][:60]}')
print(f'Ground truth: {q0["ground_truth"][:80]}')
