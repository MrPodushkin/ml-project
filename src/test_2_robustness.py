import pandas as pd
import joblib
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

random.seed(42)

# Загружаем данные
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Используем ту же подготовку данных, что при обучении
df = df.dropna(subset=["label", "message"]).copy()
df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"].str.strip().ne("")]
df = df.drop_duplicates(subset=["message"])

# Восстанавливаем тестовую выборку
_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Загружаем настоящую модель
model = joblib.load("models/spam_model.joblib")


def modify_sms(text):
    """
    Искажаем сообщение:
    заменяем некоторые буквы,
    чтобы проверить устойчивость TF-IDF.
    """
    substitutions = {
        "a": "@",
        "o": "0",
        "e": "3",
        "i": "1",
        "s": "$",
    }

    result = []

    for char in text:
        if char.lower() in substitutions and random.random() < 0.3:
            result.append(substitutions[char.lower()])
        else:
            result.append(char)

    return "".join(result)


X_original = test_df["message"].tolist()
X_modified = [modify_sms(x) for x in X_original]

y_true = test_df["label"].tolist()

# Предсказания до изменений
pred_original = model.predict(X_original)

# Предсказания после изменений
pred_modified = model.predict(X_modified)

# Сравниваем
changed = sum(
    a != b
    for a, b in zip(pred_original, pred_modified)
)

f1_before = f1_score(
    y_true,
    pred_original,
    pos_label="spam"
)

f1_after = f1_score(
    y_true,
    pred_modified,
    pos_label="spam"
)

accuracy_before = accuracy_score(
    y_true,
    pred_original
)

accuracy_after = accuracy_score(
    y_true,
    pred_modified
)

print("\n=== РЕАЛЬНЫЙ ТЕСТ УСТОЙЧИВОСТИ ===")

print("Всего SMS:", len(X_original))
print("Изменилось предсказаний:", changed)

print(f"F1 до: {f1_before:.4f}")
print(f"F1 после: {f1_after:.4f}")

print(f"Accuracy до: {accuracy_before:.4f}")
print(f"Accuracy после: {accuracy_after:.4f}")

print("\n=== ПРИМЕРЫ ИЗМЕНЕНИЙ ===")

shown = 0

for original, modified, before, after in zip(
    X_original,
    X_modified,
    pred_original,
    pred_modified
):
    if before != after:
        print("\nОригинал:", original[:150])
        print("Изменённый:", modified[:150])
        print("До:", before)
        print("После:", after)

        shown += 1

        if shown >= 5:
            break

if shown == 0:
    print("Изменений предсказаний не обнаружено.")
