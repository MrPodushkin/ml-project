"""One verified dataset and one deterministic, deduplicated split."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/SMSSpamCollection"
DATA_HASH_PATH = ROOT / "data/dataset.sha256"
SEED = 42
TEST_FRACTION = 0.2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_keys(messages):
    return messages.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def load_and_check(path=DATA_PATH):
    path = Path(path)
    raw_hash = sha256_file(path)
    expected = DATA_HASH_PATH.read_text(encoding="utf-8").split()[0]
    if raw_hash != expected:
        raise ValueError("SHA256 датасета отличается от эталона ЛР1")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )
    original_count = len(df)
    missing = int(df[["label", "message"]].isna().sum().sum())
    empty = int(df.message.str.strip().eq("").sum())
    bad_labels = sorted(set(df.label) - {"ham", "spam"})
    if missing or empty or bad_labels:
        raise ValueError("Некорректные пропуски, пустые сообщения или метки")
    df["duplicate_key"] = normalized_keys(df.message)
    conflicts = df.groupby("duplicate_key").label.nunique()
    bad_keys = set(conflicts[conflicts > 1].index)
    conflicting_rows = int(df.duplicate_key.isin(bad_keys).sum())
    df = df[~df.duplicate_key.isin(bad_keys)]
    before = len(df)
    df = df.drop_duplicates(subset="duplicate_key").copy()
    duplicates = before - len(df)
    df = df.drop(columns="duplicate_key").reset_index(drop=True)
    return df, {
        "dataset_path": "data/SMSSpamCollection",
        "dataset_sha256": raw_hash,
        "reference": "data/dataset.sha256",
        "original_rows": original_count,
        "duplicates_removed": duplicates,
        "conflicting_label_rows_removed": conflicting_rows,
        "remaining_rows": len(df),
        "class_counts": {str(k): int(v) for k, v in df.label.value_counts().items()},
        "missing_values": missing,
        "empty_messages": empty,
        "label_values": sorted(set(df.label)),
        "feature_types": {"message": "string", "label": "string"},
    }


def split_dataset(df):
    train, test = train_test_split(
        df, test_size=TEST_FRACTION, random_state=SEED, stratify=df.label
    )
    if set(normalized_keys(train.message)) & set(normalized_keys(test.message)):
        raise ValueError("Пересечение train и test по нормализованному тексту")
    return train, test


def split_fingerprints(train, test):
    return {
        name: hashlib.sha256(
            part[["label", "message"]].to_csv(index=True).encode("utf-8")
        ).hexdigest()
        for name, part in [("train", train), ("test", test)]
    }
