
import joblib

model = joblib.load("models/spam_model.joblib")

messages = [
    "Hi, are we meeting at 6 tonight?",
    "Congratulations! You won a free prize. Claim now!",
    "Please call me when you arrive.",
    "URGENT! You have won cash. Click to claim."
]

for message in messages:
    prediction = model.predict([message])[0]
    print(f"Сообщение: {message}")
    print(f"Предсказание: {prediction}")
    print("-" * 40)

