"""Evaluate a persisted real classifier on the immutable CI control set."""

import json

import joblib
from sklearn.dummy import DummyClassifier

from ci.common import REPORTS, read_json, scenario, write_json
from ci.data_gate import check
from src.data_utils import ROOT, load_and_check, sha256_file, split_dataset
from src.model_utils import MODEL_PATH, load_model
from src.train import score


def main():
    policy = read_json(ROOT / "config/data_policy.json")
    check(REPORTS / "control.json", policy)
    rows = read_json(REPORTS / "control.json")["records"]
    messages = [row["message"] for row in rows]
    labels = [row["label"] for row in rows]
    model, model_path = load_model(), MODEL_PATH
    kind = "saved Lab 2 Pipeline"
    if scenario() == "model":
        # A real, deliberately weak fitted estimator; not a mock or fabricated metric.
        weak = DummyClassifier(strategy="constant", constant="ham")
        frame, _ = load_and_check()
        train, _ = split_dataset(frame)
        weak.fit(train[["message"]], train.label)
        model_path = REPORTS / "weak_candidate.joblib"
        joblib.dump(weak, model_path)
        model = joblib.load(model_path)
        predictions = model.predict([[message] for message in messages])
        kind = "real fitted constant-ham DummyClassifier demonstration"
    else:
        predictions = model.predict(messages)
    metrics = score(labels, predictions)
    result = {
        "metric": "f1_spam",
        "value": metrics["f1_spam"],
        "metrics": metrics,
        "data_version": policy["version"],
        "data_sha256": sha256_file(REPORTS / "control.json"),
        "source_sha256": policy["source_sha256"],
        "model_sha256": sha256_file(model_path),
        "sample_size": len(rows),
        "candidate_kind": kind,
    }
    write_json(REPORTS / "candidate-metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
