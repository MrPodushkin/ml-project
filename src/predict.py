"""Validated public prediction interface."""
from pathlib import Path
import joblib

MAX_SMS_CHARS = 1000
MODEL_PATH = Path('models/spam_pipeline.joblib')


class InputValidationError(ValueError):
    """Expected invalid input (not a model failure)."""


def validate_message(message, max_chars=MAX_SMS_CHARS):
    if not isinstance(message, str):
        raise InputValidationError('Ожидается строка SMS, а не число, список или объект')
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError('max_chars должен быть положительным целым числом')
    if not message.strip():
        raise InputValidationError('Пустое SMS запрещено')
    if len(message) > max_chars:
        raise InputValidationError(f'Длина SMS превышает лимит {max_chars} символов')
    if '\x00' in message:
        raise InputValidationError('Нулевые байты запрещены')
    try:
        message.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise InputValidationError('Некорректный Unicode в сообщении') from exc
    return message


def predict_sms(message, model, max_chars=MAX_SMS_CHARS):
    # Crucial: validation precedes model.predict.
    clean = validate_message(message, max_chars=max_chars)
    return str(model.predict([clean])[0])


def load_model(path=MODEL_PATH):
    # joblib/pickle may run code during deserialization. Load only local,
    # trusted artifacts created by this project; never load user uploads.
    return joblib.load(path)
