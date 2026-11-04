"""Verify the canonical artifact, then predict using the real Pipeline."""

from src.data_utils import sha256_file
from src.model_utils import MODEL_PATH, load_model
from src.predict_sms import predict_sms


def main():
    model = load_model()
    print("Model:", MODEL_PATH.name)
    print("SHA256:", sha256_file(MODEL_PATH))
    print("Real prediction:", predict_sms("Hello, how are you?", model))


if __name__ == "__main__":
    main()
