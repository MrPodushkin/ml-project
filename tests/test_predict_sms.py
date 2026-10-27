
import pytest
from unittest.mock import Mock

from src.predict_sms import (
    predict_sms,
    InputValidationError
)


@pytest.fixture
def mock_model():
    """Поддельная модель для проверки функции."""
    model = Mock()
    model.predict.return_value = ["ham"]
    return model


@pytest.mark.parametrize(
    "message",
    [
        None,
        12345,
        "",
        "   ",
        float("nan"),
        float("inf"),
        "A" * 1001,
        "Hello\x00world",
    ]
)
def test_invalid_sms(message, mock_model):
    """Некорректные SMS должны отклоняться."""

    with pytest.raises(InputValidationError):
        predict_sms(message, mock_model)

    # Самое важное: модель не должна вызываться!
    mock_model.predict.assert_not_called()


def test_valid_sms(mock_model):
    """Корректное SMS должно обрабатываться."""

    message = "Hello, how are you?"

    result = predict_sms(message, mock_model)

    assert result == "ham"

    mock_model.predict.assert_called_once_with(
        [message]
    )

