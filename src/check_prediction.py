try:
    from src.predict_sms import InputValidationError, load_model, predict_sms
except ModuleNotFoundError:
    from predict_sms import InputValidationError, load_model, predict_sms

model = load_model()

messages = [
    "Hello, how are you?",
    "Call",
    None,
    12345,
    "A" * 1500,
    "\x00Invalid message",
    "Привет!  Киберустойчивость",
]

for i, message in enumerate(messages, 1):
    print(f"\nТест {i}")
    print("Вход:", repr(message)[:100])

    try:
        result = predict_sms(message, model)
        print("Предсказание:", result)

    except InputValidationError as error:
        print("Отклонено:", error)
