
import hashlib
from pathlib import Path

import joblib

from src.predict_sms import predict_sms

MODEL_PATH = Path("models/spam_model.joblib")
HASH_PATH = Path("model.sha256")


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Файл модели отсутствует")

    current_hash = calculate_sha256(MODEL_PATH)

    print("SHA256 модели:", current_hash)

    # Проверяем сохранённую контрольную сумму
    expected_hash = HASH_PATH.read_text(
        encoding="utf-8"
    ).split()[0]

    if current_hash != expected_hash:
        raise ValueError("Контрольная сумма модели не совпадает")

    print("Контроль целостности: OK")

    # Загружаем только собственную доверенную модель
    model = joblib.load(MODEL_PATH)

    messages = [
        "Hello, how are you?",
        "Congratulations! Claim your free prize now!"
    ]

    for message in messages:
        prediction = predict_sms(message, model)

        print("\nSMS:", message)
        print("Предсказание:", prediction)

    print("\nМодель успешно загружена и работает")


if __name__ == "__main__":
    main()
