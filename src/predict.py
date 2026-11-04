"""Compatibility names delegating to the single validated implementation."""

from src.predict_sms import MAX_MESSAGE_LENGTH as MAX_SMS_CHARS
from src.predict_sms import InputValidationError, load_model, predict_sms, validate_sms

__all__ = ["InputValidationError", "load_model", "predict_sms", "validate_message", "MAX_SMS_CHARS"]


def validate_message(message, max_chars=MAX_SMS_CHARS):
    return validate_sms(message, max_chars)
