
from pathlib import Path
import joblib

MAX_MESSAGE_LENGTH = 1000


class InputValidationError(ValueError):
    """Ошибка проверки входного SMS."""


def validate_sms(message):
    """
    Проверяет входное SMS и возвращает корректную строку.
    """

    # 1. Проверка типа
    if not isinstance(message, str):
        raise InputValidationError(
            "SMS должно быть строкой (str)."
        )

    # 2. Проверка пустого сообщения
    if not message.strip():
        raise InputValidationError(
            "SMS не должно быть пустым."
        )

    # 3. Проверка длины
    if len(message) > MAX_MESSAGE_LENGTH:
        raise InputValidationError(
            f"SMS превышает ограничение "
            f"{MAX_MESSAGE_LENGTH} символов."
        )

    # 4. Проверка нулевых байтов
    if "\x00" in message:
        raise InputValidationError(
            "SMS содержит запрещённый нулевой байт."
        )

    # 5. Проверка некорректного Unicode
    try:
        message.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputValidationError(
            "SMS содержит некорректные Unicode-символы."
        ) from exc

    return message


def predict_sms(message, model):
    """
    Проверяет SMS, выполняет предобработку
    и возвращает предсказание модели.
    """

    clean_message = validate_sms(message)

    # TF-IDF уже включён в обученный Pipeline.
    prediction = model.predict([clean_message])

    return str(prediction[0])


def load_model(
    model_path="models/spam_model.joblib"
):
    """
    Загружает локальную проверенную модель.
    """
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Файл модели не найден: {path}"
        )

    return joblib.load(path)

