"""Capture non-secret provenance only; never dump environment variables."""

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

from ci.common import REPORTS, write_json
from src.data_utils import ROOT, sha256_file


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "commit": commit,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "tools": {name: version(name) for name in ("pytest", "ruff", "bandit", "pip-audit")},
        "runtime_versions": {
            name: version(name)
            for name in (
                "scikit-learn",
                "numpy",
                "scipy",
                "pandas",
                "joblib",
                "fastapi",
                "uvicorn",
                "starlette",
                "prometheus-client",
            )
        },
        "policy_sha256": {
            name: sha256_file(ROOT / "config" / name)
            for name in (
                "data_policy.json",
                "model_policy.json",
                "model_baseline.json",
                "tools.json",
            )
        },
    }
    write_json(REPORTS / "run-metadata.json", result)
    print(result)


if __name__ == "__main__":
    main()
