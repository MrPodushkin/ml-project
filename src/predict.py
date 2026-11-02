"""Совместимый интерфейс предсказания для старых модулей проекта."""

from src.predict_sms import (
    InputValidationError,
    load_model,
    predict_sms as _predict_sms,
    validate_sms,
)


MAX_SMS_CHARS = 1000


def validate_message(message, max_chars=MAX_SMS_CHARS):
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars < 1
    ):
        raise ValueError("max_chars должен быть положительным целым числом")

    if not isinstance(message, str):
        raise InputValidationError("Ожидается строка SMS")

    if len(message) > max_chars:
        raise InputValidationError(
            f"Длина SMS превышает лимит {max_chars} символов"
        )

    return validate_sms(message)


def predict_sms(message, model, max_chars=MAX_SMS_CHARS):
    clean_message = validate_message(message, max_chars)
    return _predict_sms(clean_message, model)
