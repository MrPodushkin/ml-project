
try:
    from src.predict_sms import (
        predict_sms,
        load_model,
        InputValidationError
    )
except ModuleNotFoundError:
    from predict_sms import (
        predict_sms,
        load_model,
        InputValidationError
    )

model = load_model()

messages = [
    "Hello, how are you?",
    "Congratulations! You won a free prize!",
    "",
    None,
    12345,
    "A" * 1500,
    "\x00Invalid message",
]

for i, message in enumerate(messages, 1):
    print(f"\nТест {i}")
    print("Вход:", repr(message)[:100])

    try:
        result = predict_sms(message, model)
        print("Предсказание:", result)

    except InputValidationError as error:
        print("Отклонено:", error)

