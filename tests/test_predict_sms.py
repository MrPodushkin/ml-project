"""Integration tests: real persisted sklearn Pipeline, no mocks or method replacement."""

import sys
from contextlib import contextmanager

import pytest

from src.predict_sms import InputValidationError, load_model, predict_sms


@pytest.fixture(scope="session")
def real_model():
    return load_model()


@contextmanager
def actual_predict_calls(model):
    # Observe the real Python call stack. Pipeline.predict is never substituted.
    calls = []
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_name == "predict"
            and frame.f_locals.get("self") is model
        ):
            calls.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize(
    "message",
    [
        None,
        12345,
        [],
        {"message": "hello"},
        "",
        "   ",
        float("nan"),
        float("inf"),
        "A" * 1001,
        "Hello\x00world",
        "\ud800",
    ],
    ids=[
        "none",
        "integer",
        "list",
        "dict",
        "empty",
        "whitespace",
        "nan",
        "inf",
        "oversized",
        "nul",
        "invalid_unicode",
    ],
)
def test_invalid_inputs_never_reach_real_model(message, real_model):
    with actual_predict_calls(real_model) as calls:
        with pytest.raises(InputValidationError):
            predict_sms(message, real_model)
    assert calls == []


@pytest.mark.parametrize(
    "message",
    [
        "Hello, how are you?",
        "See you tomorrow at 5 PM.",
        "Congratulations! Claim your free prize now!",
        "🎁 Special offer!",
        "Привет! Как дела?",
    ],
)
def test_valid_inputs_call_real_model_once(message, real_model):
    with actual_predict_calls(real_model) as calls:
        result = predict_sms(message, real_model)
    assert calls == ["predict"]
    assert result in ("ham", "spam")
    assert result == real_model.predict([message])[0]


def test_deterministic_real_prediction(real_model):
    assert predict_sms("See you tomorrow", real_model) == predict_sms(
        "See you tomorrow", real_model
    )


def test_real_model_boundary(real_model):
    assert predict_sms("A" * 1000, real_model) in ("ham", "spam")
    with actual_predict_calls(real_model) as calls:
        with pytest.raises(InputValidationError):
            predict_sms("A" * 1001, real_model)
    assert not calls


def test_small_artificial_limit_with_real_model(real_model):
    assert predict_sms("A" * 20, real_model, max_chars=20) in ("ham", "spam")
    with actual_predict_calls(real_model) as calls:
        with pytest.raises(InputValidationError):
            predict_sms("A" * 21, real_model, max_chars=20)
    assert not calls


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        load_model("models/nonexistent_model.joblib")


def test_corrupt_real_artifact_rejected_before_deserialization(tmp_path):
    corrupt = tmp_path / "spam_pipeline.joblib"
    corrupt.write_bytes(b"not the trained artifact")
    with pytest.raises(ValueError, match="Контрольная сумма"):
        load_model(corrupt)
