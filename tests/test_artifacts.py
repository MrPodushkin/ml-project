"""Verify real data, saved model, recorded metrics and the experiment's provenance."""

import json

import pytest
from sklearn.metrics import f1_score

from src.data_utils import ROOT, load_and_check, normalized_keys, sha256_file, split_dataset
from src.evaluate_model import load_experiment
from src.model_utils import MODEL_PATH
from src.robustness import small_typo


def test_real_dataset_integrity_and_no_normalized_overlap():
    df, report = load_and_check()
    train, test = split_dataset(df)
    assert report["original_rows"] == 5572
    assert len(df) == 5157
    assert len(train) == 4125 and len(test) == 1032
    assert not set(normalized_keys(train.message)) & set(normalized_keys(test.message))


def test_loaded_real_model_reproduces_baseline():
    _, test, model, baseline = load_experiment()
    predicted = model.predict(test.message.astype(str))
    assert f1_score(test.label, predicted, pos_label="spam") == pytest.approx(
        baseline["primary_metric_value"]
    )
    assert sha256_file(MODEL_PATH) == baseline["model_sha256"]


def test_robustness_uses_real_model_test_only():
    train, test, _, baseline = load_experiment()
    report = json.loads((ROOT / "metrics/robustness.json").read_text())
    assert report["sample_size"] == 20
    assert len(set(report["sample_ids"])) == 20
    assert set(report["sample_ids"]) <= set(test.index)
    assert not set(report["sample_ids"]) & set(train.index)
    assert report["sample_train_overlap"] == 0
    assert report["model_sha256"] == baseline["model_sha256"]
    assert report["split_fingerprints"] == baseline["split_fingerprints"]


def test_small_typo_changes_only_one_character():
    original = "Congratulations! Claim your free prize now!"
    modified = small_typo(original)
    assert len(modified) == len(original)
    assert sum(a != b for a, b in zip(original, modified)) == 1
