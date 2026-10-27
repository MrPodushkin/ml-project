
import time
import json
import sys
from pathlib import Path

import pandas as pd
import sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Параметры эксперимента
SEED = 42
TEST_SIZE = 0.2

# Создаём папки
Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

# 1. Загружаем исходный датасет
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# 2. Очистка (та же логика, что в пункте 2)
df = df.dropna(subset=["label", "message"]).copy()
df = df[df["label"].isin(["ham", "spam"])].copy()
df = df[df["message"].str.strip().ne("")].copy()

# Запрещаем противоречивые метки одного текста
conflicts = df.groupby("message")["label"].nunique()
if (conflicts > 1).any():
    raise ValueError("Найдены одинаковые тексты с разными метками")

# Удаляем дубликаты до разделения
df = df.drop_duplicates(subset=["message"])

X = df["message"]
y = df["label"]

# 3. Делим данные так же, как в пункте 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

# 4. Создаём Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=SEED
    ))
])

# 5. Обучаем модель и измеряем время
start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start

# 6. Предсказание на тестовых данных
y_pred = model.predict(X_test)

# 7. Простая базовая модель: всегда самый частый класс
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

def calculate_metrics(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_spam": precision_score(
            y_true, predictions, pos_label="spam", zero_division=0
        ),
        "recall_spam": recall_score(
            y_true, predictions, pos_label="spam", zero_division=0
        ),
        "f1_spam": f1_score(
            y_true, predictions, pos_label="spam", zero_division=0
        )
    }

model_metrics = calculate_metrics(y_test, y_pred)
dummy_metrics = calculate_metrics(y_test, dummy_pred)

# 8. Результаты
print("=== ОБУЧЕНИЕ МОДЕЛИ ===")
print("Количество объектов после очистки:", len(df))
print("Train:", len(X_train))
print("Test:", len(X_test))
print("Время обучения:", round(training_time, 4), "сек.")
print("Seed:", SEED)

print("\n=== LOGISTIC REGRESSION ===")
for name, value in model_metrics.items():
    print(f"{name}: {value:.4f}")

print("\n=== DUMMY CLASSIFIER ===")
for name, value in dummy_metrics.items():
    print(f"{name}: {value:.4f}")

print("\n=== МАТРИЦА ОШИБОК ===")
print(confusion_matrix(
    y_test, y_pred, labels=["ham", "spam"]
))

print("\n=== ВЕРСИИ ===")
print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)

# 9. Сохранение обученной модели
joblib.dump(model, "models/spam_model.joblib")

# 10. Сохранение параметров и метрик
results = {
    "model": "TF-IDF + LogisticRegression",
    "seed": SEED,
    "test_size": TEST_SIZE,
    "training_time_seconds": training_time,
    "train_size": len(X_train),
    "test_size_objects": len(X_test),
    "parameters": {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "max_iter": 1000
    },
    "library_versions": {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__
    },
    "logistic_regression": model_metrics,
    "dummy_classifier": dummy_metrics,
    "confusion_matrix": confusion_matrix(
        y_test, y_pred, labels=["ham", "spam"]
    ).tolist()
}

with open("metrics/training_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

print("\nМодель сохранена: models/spam_model.joblib")
print("Метрики сохранены: metrics/training_results.json")
