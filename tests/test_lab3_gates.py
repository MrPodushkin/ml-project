"""Behavioral gate tests using actual data copies and real model objects; no mocks."""

from copy import deepcopy
from importlib.metadata import requires, version

import pandas as pd
import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ci.common import REPORTS, read_json
from ci.data_gate import check, validate_control
from ci.model_gate import compare
from ci.security_gate import inspect_report
from src.data_utils import ROOT, normalized_keys
from src.model_utils import load_model
from src.train import score


@pytest.fixture
def control():
    return read_json(REPORTS / "control.json")


@pytest.fixture
def data_policy():
    return read_json(ROOT / "config/data_policy.json")


def test_preprocessing_normalization_and_shape():
    raw = pd.Series(["  HELLO   world  ", "hello world", "Free\tprize"], dtype="string")
    assert normalized_keys(raw).tolist() == ["hello world", "hello world", "free prize"]
    vectorizer = load_model().named_steps["tfidf"]
    result = vectorizer.transform(raw)
    assert result.shape == (3, len(vectorizer.vocabulary_))
    assert (result[0] != result[1]).nnz == 0


def test_metric_known_confusion_counts():
    # TP=1, FP=1, FN=1, TN=1: all four headline metrics equal 1/2.
    result = score(["spam", "spam", "ham", "ham"], ["spam", "ham", "spam", "ham"])
    for metric in ("accuracy", "precision_spam", "recall_spam", "f1_spam"):
        assert result[metric] == pytest.approx(0.5)


@pytest.mark.parametrize("filename", ["requirements-runtime.txt", "requirements-service.txt"])
def test_runtime_lock_covers_actual_transitive_dependencies(filename):
    pins = {}
    for line in (ROOT / filename).read_text().splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            assert str(requirement.specifier).startswith("==")
            pins[canonicalize_name(requirement.name)] = requirement
    for name, pin in pins.items():
        assert version(name) in pin.specifier
        for dependency in requires(name) or []:
            required = Requirement(dependency)
            if required.marker is None or required.marker.evaluate({"extra": ""}):
                assert canonicalize_name(required.name) in pins, (
                    f"Unpinned transitive dependency: {required.name}"
                )


def test_verified_control_passes(data_policy):
    result = check(REPORTS / "control.json", data_policy)
    assert result["passed"]
    assert result["ham"] == result["spam"] == 10


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), [], 7, "", " ", "A" * 1001])
def test_missing_or_invalid_sms_is_rejected_not_filled(control, data_policy, bad):
    control["records"][0]["message"] = bad
    with pytest.raises((ValueError, TypeError)):
        validate_control(control, data_policy)


@pytest.mark.parametrize("mutation", ["schema", "label", "id", "version", "balance", "duplicate"])
def test_data_policy_violations(control, data_policy, mutation):
    if mutation == "schema":
        del control["records"][0]["message"]
    elif mutation == "label":
        control["records"][0]["label"] = "other"
    elif mutation == "id":
        control["records"][0]["row_id"] = True
    elif mutation == "version":
        control["version"] = "unexpected-v2"
    elif mutation == "balance":
        for row in control["records"]:
            row["label"] = "ham"
    else:
        control["records"][1]["message"] = control["records"][0]["message"]
    with pytest.raises(ValueError):
        validate_control(control, data_policy)


def test_byte_change_rejected_even_if_semantically_equal(tmp_path, data_policy):
    changed = tmp_path / "changed.json"
    changed.write_bytes((REPORTS / "control.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        check(changed, data_policy)


def baseline_candidate():
    baseline = read_json(ROOT / "config/model_baseline.json")
    candidate = {**baseline, "model_sha256": baseline["lab2_model_sha256"]}
    policy = read_json(ROOT / "config/model_policy.json")
    return candidate, baseline, policy


def test_model_threshold_accepts_boundary_and_rejects_below():
    candidate, baseline, policy = baseline_candidate()
    candidate["value"] = baseline["value"] * 0.95
    assert compare(candidate, baseline, policy)["passed"]
    candidate["value"] -= 0.000001
    assert not compare(candidate, baseline, policy)["passed"]


@pytest.mark.parametrize("value", [None, "0.99", True, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metric_fails_closed(value):
    candidate, baseline, policy = baseline_candidate()
    candidate["value"] = value
    with pytest.raises(ValueError):
        compare(candidate, baseline, policy)


def test_missing_metric_and_incompatible_data_fail_closed():
    candidate, baseline, policy = baseline_candidate()
    del candidate["value"]
    with pytest.raises(KeyError):
        compare(candidate, baseline, policy)
    candidate["value"] = 1.0
    candidate["data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data_sha256"):
        compare(candidate, baseline, policy)


def test_security_policy_blocks_simulated_critical_and_unknown():
    report = read_json(ROOT / "tests/fixtures/trivy-critical-simulation.json")
    assert not inspect_report(report)["passed"]
    unknown = deepcopy(report)
    unknown["Results"][0]["Vulnerabilities"][0].pop("Severity")
    assert not inspect_report(unknown)["passed"]
    invalid = deepcopy(report)
    invalid["Results"][0]["Vulnerabilities"][0]["Severity"] = "UNRECOGNIZED"
    with pytest.raises(ValueError):
        inspect_report(invalid)


def test_security_rejects_missing_results_and_accepts_completed_clean_scan():
    with pytest.raises(ValueError):
        inspect_report({"SchemaVersion": 2})
    assert inspect_report(
        {"SchemaVersion": 2, "Results": [{"Target": "Debian runtime", "Vulnerabilities": []}]}
    )["passed"]
