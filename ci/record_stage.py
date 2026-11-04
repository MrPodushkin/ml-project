"""Record local stage status and explicit skipped downstream stages."""

import sys

from ci.common import REPORTS, read_json, write_json

STAGES = [
    "artifacts",
    "lint",
    "bandit",
    "tests",
    "dependency_audit",
    "security_demonstration",
    "data_gate",
    "evaluate",
    "model_gate",
    "install_scanner",
    "docker_build",
    "config_scan",
    "image_export",
    "image_scan",
    "security_gate",
    "non_root",
    "smoke",
]


def main():
    name, status = sys.argv[1:3]
    path = REPORTS / "pipeline-stages.json"
    if name == "initialize":
        results = {stage: "not_run" for stage in STAGES}
    else:
        results = read_json(path)
        if name not in results or status not in {"running", "passed", "failed"}:
            raise ValueError("Unknown stage or state")
        results[name] = status
        if status == "failed":
            for stage in STAGES[STAGES.index(name) + 1 :]:
                if results[stage] == "not_run":
                    results[stage] = "skipped"
    write_json(path, results)


if __name__ == "__main__":
    main()
