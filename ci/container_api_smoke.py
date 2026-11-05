"""Start the actual image with ephemeral runtime secrets and test its HTTP API."""

import argparse
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from ci.common import REPORTS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="ml-project:lab4")
    parser.add_argument("--sudo", action="store_true")
    args = parser.parse_args()
    if args.sudo:
        # Authenticate under this Python parent before its docker subprocesses.
        subprocess.run(["sudo", "-v"], check=True)
    docker = ["sudo", "-n", "docker"] if args.sudo else ["docker"]
    name = "ml-project-lab4-smoke-" + secrets.token_hex(4)
    with tempfile.TemporaryDirectory(prefix="lab4-smoke-") as folder:
        secret_file = Path(folder) / "runtime.env"
        secret_file.touch(mode=0o600)
        secret_file.write_text("LAB4_API_TOKEN=" + secrets.token_urlsafe(32) + "\n")
        try:
            subprocess.run(
                [
                    *docker,
                    "run",
                    "--detach",
                    "--name",
                    name,
                    "--env-file",
                    str(secret_file),
                    "--read-only",
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "64",
                    "--memory",
                    "768m",
                    "--cpus",
                    "1",
                    args.image,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            for _ in range(40):
                health = subprocess.run(
                    [*docker, "exec", name, "python", "-m", "src.service_health"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if health.returncode == 0:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Container never became healthy")
            proc = subprocess.run(
                [*docker, "exec", name, "python", "-m", "src.service_smoke"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            result = json.loads(proc.stdout)
            write_json(REPORTS / "container-smoke.json", result)
            print(json.dumps(result))
            logs = subprocess.run(
                [*docker, "logs", name], capture_output=True, text=True, timeout=10
            )
            (REPORTS / "container-api.log").write_text(logs.stdout + logs.stderr)
        finally:
            subprocess.run([*docker, "rm", "--force", name], capture_output=True, timeout=20)
            os.unlink(secret_file)


if __name__ == "__main__":
    main()
