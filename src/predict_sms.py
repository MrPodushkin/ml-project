"""Single public prediction function; validation precedes the real Pipeline."""

from src.model_utils import load_model

__all__ = [
    "InputValidationError",
    "load_model",
    "predict_sms",
    "validate_sms",
    "MAX_MESSAGE_LENGTH",
]

MAX_MESSAGE_LENGTH = 1000


class InputValidationError(ValueError):
    """Controlled rejection of an invalid SMS."""


def validate_sms(message, max_chars=MAX_MESSAGE_LENGTH):
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or not 1 <= max_chars <= MAX_MESSAGE_LENGTH
    ):
        raise ValueError("max_chars должен быть целым числом от 1 до 1000")
    if not isinstance(message, str):
        raise InputValidationError("SMS должно быть строкой (str); NaN, Inf и None отклоняются")
    if len(message) > max_chars:
        raise InputValidationError(f"SMS превышает ограничение {max_chars} символов")
    if not message.strip():
        raise InputValidationError("SMS не должно быть пустым")
    if "\x00" in message:
        raise InputValidationError("SMS содержит запрещённый нулевой байт")
    try:
        message.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputValidationError("SMS содержит некорректные Unicode-символы") from exc
    return message


def predict_sms(message, model, max_chars=MAX_MESSAGE_LENGTH):
    clean = validate_sms(message, max_chars)
    return str(model.predict([clean])[0])
