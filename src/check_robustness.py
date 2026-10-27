
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.predict_sms import predict_sms, InputValidationError


SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 1000

Path("metrics").mkdir(exist_ok=True)

# 1. Загружаем датасет
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# 2. Повторяем очистку из пунктов 2–4
df = df.dropna(subset=["label", "message"]).copy()
df = df[df["label"].isin(["ham", "spam"])].copy()
df = df[df["message"].str.strip().ne("")].copy()

conflicts = df.groupby("message")["label"].nunique()
if (conflicts > 1).any():
    raise ValueError("Обнаружены конфликтующие метки")

df = df.drop_duplicates(subset=["message"])

X = df["message"]
y = df["label"]

# 3. Воспроизводим тестовую выборку
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

test_df = pd.DataFrame({
    "message": X_test.to_numpy(),
    "label": y_test.to_numpy()
})

# 4. Небольшие изменения текста
def modify_message(message, index):
    if index % 3 == 0:
        # Меняем регистр
        return message.swapcase()

    if index % 3 == 1:
        # Добавляем один пробел
        return " " + message

    # Небольшая опечатка: перестановка двух букв
    words = message.split()
    for i, word in enumerate(words):
        if len(word) >= 5 and word.isalpha():
            words[i] = (
                word[:2] + word[3] +
                word[2] + word[4:]
            )
            return " ".join(words)

    # Если подходящего слова нет
    return message + " "


# 5. Выбираем 20 SMS: 10 ham, 10 spam
# Длину ограничиваем до отбора, а не после предсказания.
def choose_messages(label, count=10):
    candidates = test_df[
        (test_df["label"] == label) &
        (test_df["message"].str.len() <= MAX_LENGTH - 2)
    ]

    if len(candidates) < count:
        raise ValueError(
            f"Недостаточно сообщений класса {label}"
        )

    return candidates.sample(
        n=count,
        random_state=SEED
    )


sample = pd.concat([
    choose_messages("ham"),
    choose_messages("spam")
]).reset_index(drop=True)

# 6. Загружаем нашу обученную модель
# Используем только собственный доверенный файл.
model = joblib.load("models/spam_model.joblib")

rows = []

# 7. Сравниваем исходные и изменённые SMS
for index, row in sample.iterrows():
    original = row["message"]
    modified = modify_message(original, index)

    pred_before = predict_sms(original, model)
    pred_after = predict_sms(modified, model)

    rows.append({
        "number": index + 1,
        "actual": row["label"],
        "original": original,
        "modified": modified,
        "prediction_before": pred_before,
        "prediction_after": pred_after,
        "changed": pred_before != pred_after
    })

result_df = pd.DataFrame(rows)

# 8. Считаем долю изменившихся ответов
changed_count = int(result_df["changed"].sum())
changed_fraction = changed_count / len(result_df)

# 9. Сравниваем F1-score на одной группе SMS
f1_before = f1_score(
    result_df["actual"],
    result_df["prediction_before"],
    pos_label="spam",
    zero_division=0
)

f1_after = f1_score(
    result_df["actual"],
    result_df["prediction_after"],
    pos_label="spam",
    zero_division=0
)

print("=== ЭКСПЕРИМЕНТ С УСТОЙЧИВОСТЬЮ ===")
print("Сообщений:", len(result_df))
print("Изменилось предсказаний:", changed_count)
print("Доля изменений: {:.2f}%".format(
    changed_fraction * 100
))
print("F1 до:", round(f1_before, 4))
print("F1 после:", round(f1_after, 4))

# 10. Показываем результаты
print("\n=== СРАВНЕНИЕ ПРЕДСКАЗАНИЙ ===")

for _, row in result_df.iterrows():
    print("-" * 45)
    print("Номер:", row["number"])
    print("Истинная метка:", row["actual"])
    print("До:", row["prediction_before"])
    print("После:", row["prediction_after"])
    print("Ответ изменился:", row["changed"])

# 11. Проверяем нетипичный вход
unusual_message = "🎁🚀💰🔥🎉" * 10

try:
    prediction = predict_sms(unusual_message, model)
    unusual_result = {
        "status": "accepted",
        "prediction": prediction
    }
except InputValidationError as error:
    unusual_result = {
        "status": "rejected",
        "reason": str(error)
    }

print("\n=== НЕТИПИЧНЫЙ ВХОД ===")
print(unusual_result)

# 12. Сохраняем результаты
results = {
    "model": "TF-IDF + LogisticRegression",
    "seed": SEED,
    "sample_size": len(result_df),
    "ham_count": 10,
    "spam_count": 10,
    "changed_predictions": changed_count,
    "changed_fraction": float(changed_fraction),
    "f1_spam_before": float(f1_before),
    "f1_spam_after": float(f1_after),
    "f1_difference": float(f1_after - f1_before),
    "unusual_input": unusual_result
}

with open(
    "metrics/robustness.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

# Подробные тексты храним отдельно
result_df.to_csv(
    "metrics/robustness_private.csv",
    index=False
)

print("\nФайлы созданы:")
print("metrics/robustness.json")
print("metrics/robustness_private.csv")
