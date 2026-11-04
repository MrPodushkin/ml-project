"""Reject CRITICAL/UNKNOWN image findings; pip-audit itself rejects every finding."""

import argparse
import json

from ci.common import REPORTS, read_json, scenario, write_json
from src.data_utils import ROOT


def inspect_report(report):
    if not isinstance(report, dict) or report.get("SchemaVersion") != 2:
        raise ValueError("Invalid Trivy report schema")
    if not isinstance(report.get("Results"), list) or not report["Results"]:
        raise ValueError("Missing scan results; absence is not a clean scan")
    findings = []
    allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}
    for target in report["Results"]:
        if not isinstance(target, dict) or not isinstance(target.get("Target"), str):
            raise ValueError("Invalid scan target")
        for item in target.get("Vulnerabilities") or []:
            severity = item.get("Severity", "UNKNOWN")
            if severity not in allowed:
                raise ValueError("Unrecognized severity")
            if severity in {"CRITICAL", "UNKNOWN"}:
                findings.append(
                    {
                        "package": item.get("PkgName"),
                        "version": item.get("InstalledVersion"),
                        "id": item.get("VulnerabilityID"),
                        "fixed_version": item.get("FixedVersion"),
                        "severity": severity,
                    }
                )
    return {"passed": not findings, "blocked_findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=REPORTS / "trivy-image.json")
    parser.add_argument("--output", default=REPORTS / "security-gate.json")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo and scenario() != "security":
        print("Security demonstration inactive")
        return 0
    path = ROOT / "tests/fixtures/trivy-critical-simulation.json" if args.demo else args.report
    try:
        result = inspect_report(read_json(path))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result = {"passed": False, "error": str(exc)}
    result.update({"simulation": args.demo, "exit_code": 0 if result["passed"] else 1})
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
