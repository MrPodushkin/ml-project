import pytest

from src.predict import (
    InputValidationError,
    load_model,
    predict_sms,
    validate_message,
)


@pytest.fixture(scope="module")
def real_model():
    return load_model()


def test_compat_prediction(real_model):
    result = predict_sms(
        "Hello, how are you?",
        real_model,
    )

    assert result == "ham"


def test_compat_invalid_message(real_model):
    with pytest.raises(InputValidationError):
        predict_sms(None, real_model)


def test_compat_length_limit(real_model):
    with pytest.raises(InputValidationError):
        predict_sms(
            "A" * 21,
            real_model,
            max_chars=20,
        )


def test_compat_valid_message():
    result = validate_message(
        "Hello",
        max_chars=20,
    )

    assert result == "Hello"


def test_compat_invalid_limit():
    with pytest.raises(ValueError):
        validate_message("Hello", max_chars=0)
