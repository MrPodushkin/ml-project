"""Negative and positive tests: invalid inputs must not reach the model."""
from unittest.mock import Mock
import pytest
from src.predict import InputValidationError, predict_sms


def mock_model():
    model = Mock()
    model.predict.return_value = ['ham']
    return model


@pytest.mark.parametrize('bad_input', [123, None, {'message': 'test'}, ['hi']])
def test_wrong_type_never_calls_model(bad_input):
    model = mock_model()
    with pytest.raises(InputValidationError):
        predict_sms(bad_input, model)
    model.predict.assert_not_called()


def test_blank_never_calls_model():
    model = mock_model()
    with pytest.raises(InputValidationError, match='Пустое'):
        predict_sms(' \n\t ', model)
    model.predict.assert_not_called()


def test_oversized_never_calls_model():
    model = mock_model()
    with pytest.raises(InputValidationError, match='лимит'):
        predict_sms('a' * 21, model, max_chars=20)
    model.predict.assert_not_called()


def test_invalid_nul_never_calls_model():
    model = mock_model()
    with pytest.raises(InputValidationError, match='Нулевые'):
        predict_sms('abc\x00def', model)
    model.predict.assert_not_called()


def test_invalid_unicode_never_calls_model():
    model = mock_model()
    with pytest.raises(InputValidationError, match='Unicode'):
        predict_sms('hi\ud800', model)
    model.predict.assert_not_called()


def test_positive_calls_model_once():
    model = mock_model()
    assert predict_sms('Hello, see you tomorrow!', model) == 'ham'
    model.predict.assert_called_once_with(['Hello, see you tomorrow!'])
