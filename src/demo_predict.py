"""Run local predictions against the trusted saved model."""
from src.predict import load_model, predict_sms, InputValidationError


def main():
    model = load_model()
    for message in ['Hi! Are we meeting at six?', 'FREE cash prize! Claim now!!!', '']:
        try:
            print(repr(message), '=>', predict_sms(message, model))
        except InputValidationError as exc:
            print(repr(message), '=> отклонено:', exc)


if __name__ == '__main__':
    main()
