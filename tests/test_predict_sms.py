import pytest

from src.predict_sms import (
    InputValidationError,
    load_model,
    predict_sms,
)

@pytest.fixture(scope="module")
def real_model():
    """Загружаем настоящую обученную модель."""
    return load_model("models/spam_model.joblib")

@pytest.mark.parametrize("message", [
    None,
    12345,
    "",
    "   ",
    float("nan"),
    float("inf"),
    "A" * 1001,
    "Hello\x00world",
    "\ud800",
])
def test_invalid_inputs(message, real_model):
    """Некорректные SMS должны отклоняться."""
    with pytest.raises(InputValidationError):
        predict_sms(message, real_model)

@pytest.mark.parametrize("message", [
    "Hello, how are you?",
    "See you tomorrow at 5 PM.",
    "Congratulations! Claim your free prize now!",
    "🎁 Special offer!",
    "Привет! Как дела?",
])
def test_valid_inputs(message, real_model):
    """Корректные сообщения должны обрабатываться моделью."""
    prediction = predict_sms(message, real_model)
    assert prediction in ("ham", "spam")

def test_prediction_matches_pipeline(real_model):
    """Результат функции должен совпадать с моделью."""
    message = "Hello, how are you?"

    result = predict_sms(message, real_model)
    expected = real_model.predict([message])[0]

    assert result == expected

def test_prediction_is_deterministic(real_model):
    """Один текст должен давать одинаковый результат."""
    message = "See you tomorrow at 5 PM."

    first = predict_sms(message, real_model)
    second = predict_sms(message, real_model)

    assert first == second

def test_message_length_boundary(real_model):
    """Проверка границы длины SMS."""
    assert predict_sms("A" * 1000, real_model) in ("ham", "spam")

    with pytest.raises(InputValidationError):
        predict_sms("A" * 1001, real_model)

def test_missing_model():
    """При отсутствии файла модели должна возникать ошибка."""
    with pytest.raises(FileNotFoundError):
        load_model("models/nonexistent_model.joblib")

@pytest.mark.parametrize(
    "message, expected",
    [
        ("Hey, are we still meeting tonight?", "ham"),
        ("I'll call you when I get home.", "ham"),
    ],
)
def test_expected_predictions(message, expected, real_model):
    result = predict_sms(message, real_model)
    assert result == expected
