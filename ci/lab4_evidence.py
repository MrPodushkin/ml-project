"""Display compact saved evidence without private request bodies or credentials."""

import argparse
import json

from ci.common import REPORTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("view", choices=["result", "logs", "before", "after", "recovered"])
    args = parser.parse_args()
    folder = REPORTS / "lab4"
    if args.view == "result":
        print(
            json.dumps(json.loads((folder / "demo.json").read_text()), indent=2, ensure_ascii=False)
        )
    elif args.view == "logs":
        for line in (folder / "api-log.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item["route"] == "/predict":
                print(json.dumps(item, ensure_ascii=False))
    else:
        for line in (folder / f"metrics-{args.view}.prom").read_text().splitlines():
            if line.startswith(
                (
                    "sms_api_requests_total",
                    "sms_api_rejections_total",
                    "sms_api_predictions_total",
                    "sms_api_detection_active",
                )
            ):
                print(line)


if __name__ == "__main__":
    main()
