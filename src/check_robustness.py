import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.predict_sms import predict_sms


DATA_PATH = Path("data/SMSSpamCollection")
MODEL_PATH = Path("models/spam_model.joblib")
RESULTS_DIR = Path("metrics")

SEED = 42


def load_test_data():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df = df.dropna(subset=["label", "message"]).copy()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"].str.strip().ne("")]
    df = df.drop_duplicates(subset=["message"])

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    return test_df


def simple_change(text):
    """Простое изменение регистра и пробелов."""
    return " " + text.swapcase()


def strong_change(text):
    """Замена букв цифрами для проверки устойчивости."""
    replacements = {
        "a": "4",
        "e": "3",
        "i": "1",
        "o": "0",
        "s": "5",
    }

    return "".join(
        replacements.get(char.lower(), char)
        if char.isalpha()
        else char
        for char in text
    )


def evaluate(model, texts, true_labels):
    predictions = [
        predict_sms(text, model)
        for text in texts
    ]

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(true_labels, predictions),
        "f1_spam": f1_score(
            true_labels,
            predictions,
            pos_label="spam",
            zero_division=0
        )
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    model = joblib.load(MODEL_PATH)
    test_df = load_test_data()

    # Используем всю отложенную тестовую выборку.
    # Оставляем только сообщения, допустимые для predict_sms.
    test_df = test_df[
        test_df["message"].str.len().le(1000)
    ].copy()

    original = test_df["message"].tolist()
    labels = test_df["label"].tolist()

    simple = [simple_change(text) for text in original]
    strong = [strong_change(text) for text in original]

    before = evaluate(model, original, labels)
    after_simple = evaluate(model, simple, labels)
    after_strong = evaluate(model, strong, labels)

    print("\n=== ПРОВЕРКА УСТОЙЧИВОСТИ ===")
    print("Количество сообщений:", len(original))

    results = {}

    for name, modified, after in [
        ("Простые изменения", simple, after_simple),
        ("Замена букв цифрами", strong, after_strong),
    ]:
        changed = sum(
            x != y
            for x, y in zip(
                before["predictions"],
                after["predictions"]
            )
        )

        print(f"\n=== {name} ===")
        print("Изменилось предсказаний:", changed)
        print("Доля изменений:", f"{changed / len(original):.2%}")
        print("F1 до:", f"{before['f1_spam']:.4f}")
        print("F1 после:", f"{after['f1_spam']:.4f}")
        print("Accuracy до:", f"{before['accuracy']:.4f}")
        print("Accuracy после:", f"{after['accuracy']:.4f}")

        results[name] = {
            "changed_predictions": changed,
            "changed_fraction": changed / len(original),
            "f1_before": before["f1_spam"],
            "f1_after": after["f1_spam"],
            "accuracy_before": before["accuracy"],
            "accuracy_after": after["accuracy"],
        }

    # Отдельный контрольный пример, уже проверенный вручную.
    original_spam = (
        "Congratulations! You have won a cash prize! Claim now!"
    )

    modified_spam = (
        "C0NGRATULATI0NS! Y0U HAVE W0N A CASH PR1ZE! CLA1M N0W!"
    )

    prediction_before = predict_sms(original_spam, model)
    prediction_after = predict_sms(modified_spam, model)

    print("\n=== КОНТРОЛЬНЫЙ ПРИМЕР ОБХОДА ===")
    print("Исходный текст:", original_spam)
    print("Изменённый текст:", modified_spam)
    print("До:", prediction_before)
    print("После:", prediction_after)

    results["control_example"] = {
        "prediction_before": prediction_before,
        "prediction_after": prediction_after,
        "changed": prediction_before != prediction_after,
    }

    results["sample_size"] = len(original)
    results["seed"] = SEED

    output_path = RESULTS_DIR / "robustness.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nРезультаты сохранены:", output_path)


if __name__ == "__main__":
    main()
