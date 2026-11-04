"""Obtain verified public data and reproduce the exact private Lab 2 artifact."""

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import joblib

from ci.common import REPORTS, read_json, write_json
from src.data_utils import DATA_PATH, ROOT, load_and_check, sha256_file, split_dataset
from src.model_utils import MODEL_PATH, load_model
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
    if not MODEL_PATH.exists():
        train, _ = split_dataset(df)
        model = build_pipeline()
        model.fit(train.message.astype(str), train.label.astype(str))
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=MODEL_PATH.parent) as folder:
            candidate = Path(folder) / "candidate.joblib"
            joblib.dump(model, candidate)
            if sha256_file(candidate) != baseline["model_sha256"]:
                raise ValueError(
                    "Rebuilt artifact differs from Lab 2 SHA256; baseline is unchanged"
                )
            os.replace(candidate, MODEL_PATH)
    load_model()
    if sha256_file(MODEL_PATH) != baseline["model_sha256"]:
        raise ValueError("Unexpected model version")
    control = make_control()
    result = {
        "source": SOURCE_URL,
        "dataset_sha256": data["dataset_sha256"],
        "model_sha256": sha256_file(MODEL_PATH),
        "control_sha256": sha256_file(control),
        "baseline_sha256": sha256_file(ROOT / "config/model_baseline.json"),
        "model_provenance": "locally trained from verified data; byte-identical to Lab 2",
    }
    write_json(REPORTS / "provenance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare_artifacts()
