"""Prepare an isolated CI candidate without changing the Lab 2 reference files."""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import joblib

from ci.common import REPORTS, read_json, write_json
from src.data_utils import DATA_PATH, ROOT, load_and_check, sha256_file, split_dataset
from src.model_utils import MODEL_HASH_PATH, MODEL_PATH, load_model
from src.prepare_data import prepare
from src.train import build_pipeline

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def make_control():
    policy = read_json(ROOT / "config/data_policy.json")
    df, _ = load_and_check()
    train, test = split_dataset(df)
    ids = policy["row_ids"]
    if not set(ids) <= set(test.index) or set(ids) & set(train.index):
        raise ValueError("CI rows must belong only to the held-out test")
    records = [
        {"row_id": i, "label": str(df.loc[i, "label"]), "message": str(df.loc[i, "message"])}
        for i in ids
    ]
    control = {
        "version": policy["version"],
        "source_sha256": policy["source_sha256"],
        "records": records,
    }
    path = REPORTS / "control.json"
    write_json(path, control)
    return path


def prepare_artifacts():
    REPORTS.mkdir(exist_ok=True)
    if MODEL_PATH.resolve() != (REPORTS / "runtime/spam_pipeline.joblib").resolve():
        raise ValueError("Activate the isolated CI profile first: source ci/env.sh")
    if not DATA_PATH.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            blob = response.read(1024 * 1024 + 1)
        if len(blob) > 1024 * 1024:
            raise ValueError("Download exceeds 1 MiB")
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "sms.zip"
            archive.write_bytes(blob)
            prepare(archive)
    df, data = load_and_check()
    baseline = read_json(ROOT / "metrics/baseline.json")
    canonical = ROOT / "models/spam_pipeline.joblib"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if canonical.exists():
        if sha256_file(canonical) != baseline["model_sha256"]:
            raise ValueError("Local Lab 2 artifact changed; refusing to trust or replace it")
        shutil.copyfile(canonical, MODEL_PATH)
        origin = "verified copy of saved Lab 2 artifact"
    else:
        train, _ = split_dataset(df)
        model = build_pipeline()
        model.fit(train.message.astype(str), train.label.astype(str))
        with tempfile.TemporaryDirectory(dir=MODEL_PATH.parent) as folder:
            candidate = Path(folder) / "candidate.joblib"
            joblib.dump(model, candidate)
            os.replace(candidate, MODEL_PATH)
        origin = "locally rebuilt candidate from verified data and fixed Lab 2 parameters"
    model_hash = sha256_file(MODEL_PATH)
    MODEL_HASH_PATH.write_text(f"{model_hash}  models/spam_pipeline.joblib\n", encoding="utf-8")
    # Only bind a run-specific artifact to the unchanged expected Lab 2 metrics/split.
    # This is not a new metric baseline; neither reference file is overwritten.
    metadata = {
        **baseline,
        "model_sha256": model_hash,
        "model_file": str(MODEL_PATH.relative_to(ROOT)),
        "reference_model_sha256": baseline["model_sha256"],
    }
    write_json(ROOT / os.environ["ML_EXPERIMENT_METADATA"], metadata)
    load_model()
    control = make_control()
    from src.robustness import run

    run()
    result = {
        "source": SOURCE_URL,
        "dataset_sha256": data["dataset_sha256"],
        "model_sha256": model_hash,
        "lab2_model_sha256": baseline["model_sha256"],
        "matches_lab2_bytes": model_hash == baseline["model_sha256"],
        "control_sha256": sha256_file(control),
        "baseline_sha256": sha256_file(ROOT / "config/model_baseline.json"),
        "model_provenance": origin,
    }
    write_json(REPORTS / "provenance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare_artifacts()
