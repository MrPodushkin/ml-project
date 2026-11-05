"""Short demonstration commands; read runtime credentials without printing them."""

import argparse
import http.client
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "health",
            "predict",
            "invalid",
            "unauthorized",
            "oversized",
            "metrics",
            "detection",
        ],
    )
    parser.add_argument("--message", default="Hi! Are we meeting at six?")
    args = parser.parse_args()
    path = Path.home() / ".config" / "ml-project" / "lab4.env"
    credentials = dict(
        line.strip().split("=", 1)
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    route = {"health": "/health", "metrics": "/metrics", "detection": "/monitoring/detection"}.get(
        args.action, "/predict"
    )
    token = credentials[
        "LAB4_METRICS_TOKEN" if route in ("/metrics", "/monitoring/detection") else "LAB4_API_TOKEN"
    ]
    if args.action == "unauthorized":
        token = "incorrect-demonstration-token"
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    message = "" if args.action == "invalid" else args.message
    body = json.dumps({"message": message}) if route == "/predict" else None
    if args.action == "oversized":
        body = "A" * 8193
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)
    connection.request("POST" if route == "/predict" else "GET", route, body, headers)
    response = connection.getresponse()
    content = response.read().decode()
    print("HTTP", response.status)
    if route == "/metrics":
        print(
            "\n".join(
                line
                for line in content.splitlines()
                if line.startswith(
                    (
                        "sms_api_requests_total",
                        "sms_api_rejections_total",
                        "sms_api_predictions_total",
                        "sms_api_model_ready",
                        "sms_api_detection_active",
                    )
                )
            )
        )
    else:
        print(json.dumps(json.loads(content), indent=2, ensure_ascii=False))
    connection.close()


if __name__ == "__main__":
    main()
