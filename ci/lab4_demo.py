"""Bounded real-network detection exercise; creates before/after/recovery evidence."""

import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from ci.common import REPORTS, write_json
from src.data_utils import ROOT


def main():
    folder = REPORTS / "lab4"
    folder.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.pop("LAB4_API_TOKEN_FILE", None)
    env.pop("LAB4_METRICS_TOKEN_FILE", None)
    env.update(
        {
            "LAB4_API_TOKEN": token,
            "LAB4_METRICS_TOKEN": token,
            "LAB4_HOST": "127.0.0.1",
            "LAB4_PORT": str(port),
            "LAB4_RATE_LIMIT": "4",
            "LAB4_RATE_PERIOD": "2",
            "LAB4_DETECTION_WINDOW": "3",
        }
    )

    def request(path, message=None, auth=token):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {"Authorization": "Bearer " + auth, "Content-Type": "application/json"}
        connection.request(
            "GET" if message is None else "POST",
            path,
            body=None if message is None else json.dumps({"message": message}),
            headers=headers,
        )
        response = connection.getresponse()
        status, body = response.status, response.read().decode()
        connection.close()
        return status, body

    started = datetime.now(timezone.utc).isoformat()
    console = folder / "server-console.txt"
    with console.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.service"],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("Demo service failed to start")
                try:
                    if request("/health")[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("Demo readiness deadline exceeded")
            normal_before, _ = request("/predict", "Hi! Are we meeting at six?")
            assert normal_before == 200
            (folder / "metrics-before.prom").write_text(request("/metrics")[1])
            statuses = [
                request("/predict", "hello", auth="invalid-demo-token")[0] for _ in range(3)
            ]
            statuses.append(request("/predict", "hello")[0])
            assert statuses == [401, 401, 401, 429]
            active = json.loads(request("/monitoring/detection")[1])
            assert active["active"] is True
            (folder / "metrics-after.prom").write_text(request("/metrics")[1])
            time.sleep(3.2)
            normal_after, _ = request("/predict", "Hi! Are we meeting at six?")
            recovered = json.loads(request("/monitoring/detection")[1])
            assert normal_after == 200 and recovered["active"] is False
            (folder / "metrics-recovered.prom").write_text(request("/metrics")[1])
            result = {
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "rule": "At least 3 /predict responses with HTTP 401 or 429 in 3 seconds",
                "limiter": "4 requests per 2 seconds per client IP; one process",
                "normal_before_status": normal_before,
                "negative_statuses": statuses,
                "detected": active,
                "normal_after_status": normal_after,
                "recovered": recovered,
                "passed": True,
                "limitations": "A small local exercise; HTTP signals do not establish model attacks or accuracy loss.",
            }
            write_json(folder / "demo.json", result)
            print(json.dumps(result, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    events = []
    for line in console.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if "request_id" in item:
            events.append(item)
    serialized = "\n".join(json.dumps(event) for event in events) + "\n"
    assert token not in serialized and "invalid-demo-token" not in serialized
    (folder / "api-log.jsonl").write_text(serialized)


if __name__ == "__main__":
    main()
