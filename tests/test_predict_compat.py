"""Legacy API uses the same real canonical artifact."""
import pytest
from src.predict import InputValidationError,load_model,predict_sms,validate_message

def test_compat_prediction_matches_actual_pipeline():
    model=load_model()
    message='Hello, how are you?'
    assert predict_sms(message,model)==model.predict([message])[0]

def test_compat_invalid_message():
    with pytest.raises(InputValidationError):
        predict_sms(None,load_model())

def test_compat_length_limit():
    with pytest.raises(InputValidationError):
        predict_sms('A'*21,load_model(),max_chars=20)

def test_compat_valid_message():
    assert validate_message('Hello',max_chars=20)=='Hello'

@pytest.mark.parametrize('limit',[0,-1,True,1.5,1001])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        validate_message('Hello',max_chars=limit)
