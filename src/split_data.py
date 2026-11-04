"""Inspect the canonical split without fitting a second vectorizer."""

from src.data_utils import load_and_check, split_dataset, split_fingerprints


def main():
    df, report = load_and_check()
    train, test = split_dataset(df)
    print(report)
    print(
        {
            "train": len(train),
            "test": len(test),
            "split_fingerprints": split_fingerprints(train, test),
        }
    )


if __name__ == "__main__":
    main()
