"""Compare candidate F1 with a separately stored, immutable Lab 2 reference."""

import argparse
import json
import math

from ci.common import REPORTS, read_json, write_json
from src.data_utils import ROOT


def numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing or non-finite numeric metric")
    if not 0 <= value <= 1:
        raise ValueError("F1 must lie in [0, 1]")
    return float(value)


def compare(candidate, baseline, policy):
    for key in ("metric", "data_version", "data_sha256", "source_sha256", "sample_size"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"Incompatible {key}")
    reference, value = numeric(baseline["value"]), numeric(candidate["value"])
    if policy != {"direction": "higher_is_better", "max_relative_degradation": 0.05}:
        raise ValueError("Unexpected F1 threshold policy")
    if reference <= 0:
        raise ValueError("Positive reference F1 required for relative threshold")
    threshold = reference * (1 - policy["max_relative_degradation"])
    return {
        "passed": value >= threshold,
        "metric": "f1_spam",
        "baseline": reference,
        "candidate": value,
        "threshold": threshold,
        "relative_degradation_limit": 0.05,
        "model_sha256": candidate["model_sha256"],
        "data_sha256": candidate["data_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=REPORTS / "candidate-metrics.json")
    parser.add_argument("--report", default=REPORTS / "model-gate.json")
    args = parser.parse_args()
    try:
        result = compare(
            read_json(args.candidate),
            read_json(ROOT / "config/model_baseline.json"),
            read_json(ROOT / "config/model_policy.json"),
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result = {"passed": False, "error": str(exc)}
    result["exit_code"] = 0 if result["passed"] else 1
    write_json(args.report, result)
    print(json.dumps(result, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
