"""Fail closed on invalid schema, content, class balance, version or digest."""

import argparse
import json

from ci.common import REPORTS, read_json, write_json
from src.data_utils import ROOT, sha256_file
from src.predict_sms import validate_sms


def validate_control(document, policy):
    if not isinstance(document, dict) or set(document) != {"version", "source_sha256", "records"}:
        raise ValueError("Invalid top-level data schema")
    if (
        document["version"] != policy["version"]
        or document["source_sha256"] != policy["source_sha256"]
    ):
        raise ValueError("Incompatible dataset version")
    records = document["records"]
    if (
        not isinstance(records, list)
        or not policy["min_rows"] <= len(records) <= policy["max_rows"]
    ):
        raise ValueError("Invalid row count")
    ids, texts, spam = [], set(), 0
    for row in records:
        if not isinstance(row, dict) or set(row) != {"row_id", "label", "message"}:
            raise ValueError("Missing/extra fields in record")
        if type(row["row_id"]) is not int or row["label"] not in ("ham", "spam"):
            raise ValueError("Invalid row ID or label")
        message = validate_sms(row["message"], max_chars=policy["max_message_chars"])
        key = " ".join(message.lower().split())
        if key in texts:
            raise ValueError("Duplicate normalized message")
        texts.add(key)
        ids.append(row["row_id"])
        spam += row["label"] == "spam"
    if ids != policy["row_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Unexpected or duplicate row IDs")
    fraction = spam / len(records)
    if not policy["min_spam_fraction"] <= fraction <= policy["max_spam_fraction"]:
        raise ValueError("Class distribution outside policy")
    return {
        "rows": len(records),
        "ham": len(records) - spam,
        "spam": spam,
        "spam_fraction": fraction,
    }


def check(path, policy):
    result = validate_control(read_json(path), policy)
    if sha256_file(path) != policy["sha256"]:
        raise ValueError("CI dataset SHA256 mismatch")
    return {"passed": True, "version": policy["version"], "sha256": policy["sha256"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=REPORTS / "control.json")
    parser.add_argument("--report", default=REPORTS / "data-gate.json")
    args = parser.parse_args()
    try:
        result = check(args.data, read_json(ROOT / "config/data_policy.json"))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result = {"passed": False, "error": str(exc), "exit_code": 1}
    else:
        result["exit_code"] = 0
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
