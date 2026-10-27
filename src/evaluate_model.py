
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

SEED = 42
TEST_SIZE = 0.2
LABELS = ["ham", "spam"]

DATA_PATH = Path("data/SMSSpamCollection")
MODEL_PATH = Path("models/spam_model.joblib")
METRICS_DIR = Path("metrics")

METRICS_DIR.mkdir(exist_ok=True)

# 1. Загружаем данные
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# 2. Повторяем очистку из пункта 3
df = df.dropna(subset=["label", "message"]).copy()
df = df[df["label"].isin(LABELS)].copy()
df = df[df["message"].str.strip().ne("")].copy()

conflicts = df.groupby("message")["label"].nunique()
if (conflicts > 1).any():
    raise ValueError("Есть одинаковые тексты с разными метками")

df = df.drop_duplicates(subset=["message"])

X = df["message"]
y = df["label"]

# 3. Воспроизводим то же разделение, что при обучении
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

# 4. Загружаем собственную обученную модель
model = joblib.load(MODEL_PATH)

# 5. Предсказания
y_pred = model.predict(X_test)

print("=== ОТЧЁТ О КЛАССИФИКАЦИИ ===")
print(
    classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        zero_division=0,
        digits=4
    )
)

# 6. Метрики по классам
precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        zero_division=0
    )
)

per_class = {
    label: {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "support": int(support[i])
    }
    for i, label in enumerate(LABELS)
}

accuracy = accuracy_score(y_test, y_pred)
f1_spam = f1_score(
    y_test, y_pred,
    pos_label="spam",
    zero_division=0
)

f1_macro = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

print("Accuracy:", round(accuracy, 4))
print("F1 spam:", round(f1_spam, 4))
print("F1 macro:", round(f1_macro, 4))
print("F1 weighted:", round(f1_weighted, 4))

# 7. Матрица ошибок
cm = confusion_matrix(
    y_test, y_pred, labels=LABELS
)

print("\nМатрица ошибок:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

display.plot(cmap="Blues", values_format="d")
plt.title("Матрица ошибок SMS Spam Classifier")
plt.tight_layout()
plt.savefig(
    METRICS_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# 8. Неправильные предсказания
errors = pd.DataFrame({
    "message": X_test.to_numpy(),
    "actual": y_test.to_numpy(),
    "predicted": y_pred
})

errors = errors[
    errors["actual"] != errors["predicted"]
]

print("\nКоличество ошибок:", len(errors))
print("\nПримеры неправильных предсказаний:")

for _, row in errors.head(5).iterrows():
    print("-" * 45)
    print("Истинный класс:", row["actual"])
    print("Предсказанный класс:", row["predicted"])
    print("Сообщение:", str(row["message"])[:200])

# Сохраняем тексты ошибок локально для анализа.
errors.to_csv(
    METRICS_DIR / "errors_private.csv",
    index=False
)

# 9. SHA256 исходного датасета
data_hash = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

# 10. Сохраняем итоговые метрики
results = {
    "model": "TF-IDF + LogisticRegression",
    "primary_metric": "f1_spam",
    "primary_metric_value": float(f1_spam),
    "accuracy": float(accuracy),
    "f1_macro": float(f1_macro),
    "f1_weighted": float(f1_weighted),
    "per_class": per_class,
    "confusion_matrix": cm.tolist(),
    "labels": LABELS,
    "test_size": len(X_test),
    "random_state": SEED,
    "test_fraction": TEST_SIZE,
    "data_version": {
        "filename": str(DATA_PATH),
        "sha256": data_hash,
        "cleaned_rows": len(df)
    },
    "error_count": len(errors)
}

with open(
    METRICS_DIR / "baseline.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nРезультаты сохранены:")
print("metrics/baseline.json")
print("metrics/confusion_matrix.png")
print("metrics/errors_private.csv")

