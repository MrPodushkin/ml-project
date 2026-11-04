"""CLI prediction and smoke check used by the non-root runtime image."""

import argparse
import json
import os

from src.model_utils import load_model
from src.predict_sms import predict_sms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", default="Hello, how are you?")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    model = load_model()
    prediction = predict_sms(args.message, model)
    if args.smoke:
        if os.geteuid() == 0:
            raise RuntimeError("Smoke check requires a non-root container user")
        if predict_sms("Hello, how are you?", model) != "ham":
            raise RuntimeError("Unexpected ham prediction")
        if predict_sms("FREE cash prize! Claim now!!!", model) != "spam":
            raise RuntimeError("Unexpected spam prediction")
    print(json.dumps({"uid": os.geteuid(), "prediction": prediction, "smoke_passed": args.smoke}))


if __name__ == "__main__":
    main()
