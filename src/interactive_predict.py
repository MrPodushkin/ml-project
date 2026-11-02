from src.predict_sms import load_model, predict_sms, InputValidationError


def main():
    model = load_model("models/spam_model.joblib")

    print("=== ПРОВЕРКА SMS НА СПАМ ===")
    print("Введите сообщение для проверки.")
    print("Для выхода введите: exit\n")

    while True:
        message = input("SMS: ")

        if message.lower() == "exit":
            print("Программа завершена.")
            break

        try:
            prediction = predict_sms(message, model)

            if prediction == "spam":
                print("Результат: СПАМ\n")
            else:
                print("Результат: НЕ СПАМ\n")

        except InputValidationError as error:
            print(f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
