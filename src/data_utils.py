"""Load and inspect SMS Spam Collection without executing dataset-supplied scripts."""
from pathlib import Path
import hashlib
import pandas as pd

DATA_PATH = Path('data/SMSSpamCollection')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_and_check(path: Path = DATA_PATH):
    if not path.is_file():
        raise FileNotFoundError(f'Датасет не найден: {path}')
    raw_hash = sha256_file(path)
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['label', 'message'], encoding='utf-8',
                     dtype='string', keep_default_na=False)
    original_count = len(df)
    missing = int(df[['label', 'message']].isna().sum().sum())
    empty = int(df['message'].str.strip().eq('').sum())
    bad_labels = sorted(set(df['label'].dropna()) - {'ham', 'spam'})
    if missing or empty or bad_labels:
        raise ValueError(f'Ошибка данных: пропуски={missing}, пустые={empty}, метки={bad_labels}')
    # Remove duplicate SMS text, including case/whitespace variants, BEFORE splitting.
    # This reduces the chance that the same SMS leaks into both train and test.
    df['duplicate_key'] = df['message'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    conflicts = df.groupby('duplicate_key')['label'].nunique()
    conflicting_keys = set(conflicts[conflicts > 1].index)
    conflicting_rows = int(df['duplicate_key'].isin(conflicting_keys).sum())
    df = df[~df['duplicate_key'].isin(conflicting_keys)]
    before_duplicates = len(df)
    df = df.drop_duplicates(subset='duplicate_key', keep='first').copy()
    duplicate_rows = before_duplicates - len(df)
    df = df.drop(columns=['duplicate_key']).reset_index(drop=True)
    counts = {str(k): int(v) for k, v in df['label'].value_counts().items()}
    return df, {
        'dataset_path': str(path), 'dataset_sha256': raw_hash,
        'original_rows': original_count, 'duplicates_removed': duplicate_rows,
        'conflicting_label_rows_removed': conflicting_rows,
        'remaining_rows': len(df), 'class_counts': counts,
        'missing_values': missing, 'empty_messages': empty,
        'label_values': sorted(counts), 'feature_types': {'message': 'string', 'label': 'string'}
    }
