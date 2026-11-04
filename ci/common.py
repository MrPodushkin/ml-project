"""Strict JSON and report helpers shared by gates."""

import json
from pathlib import Path

from src.data_utils import ROOT

REPORTS = ROOT / "ci-artifacts"


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def scenario():
    name = read_json(ROOT / "config/ci_scenario.json")["scenario"]
    if name not in {"success", "data", "model", "security"}:
        raise ValueError("Unknown demonstration scenario")
    return name
