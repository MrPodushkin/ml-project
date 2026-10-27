
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Загружаем исходный датасет
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Исходный размер:", df.shape)

# 2. Удаляем некорректные строки
df = df.dropna(subset=["label", "message"]).copy()
df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"].str.strip() != ""]

# 3. Проверяем одинаковые тексты с разными метками
conflicts = df.groupby("message")["label"].nunique()
if (conflicts > 1).any():
    raise ValueError(
        "Есть одинаковые SMS с разными метками. "
        "Нужна ручная проверка."
    )

# 4. Удаляем повторяющиеся сообщения ДО разделения
df = df.drop_duplicates(subset=["message"])

print("После очистки:", df.shape)

# 5. Определяем признаки и целевую переменную
X = df["message"]
y = df["label"]

# 6. Разделяем на обучение и тест (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nРазмеры выборок:")
print("Обучающая:", len(X_train))
print("Тестовая:", len(X_test))

print("\nКлассы в обучающей выборке (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nКлассы в тестовой выборке (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))

# 7. Обучаем TF-IDF ТОЛЬКО на train
tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)

# 8. Применяем обученное преобразование к test
X_test_tfidf = tfidf.transform(X_test)

print("\nРазмер матриц TF-IDF:")
print("Train:", X_train_tfidf.shape)
print("Test:", X_test_tfidf.shape)

# 9. Проверяем пересечения исходных сообщений
overlap = set(X_train) & set(X_test)
print("\nПересечений между train и test:", len(overlap))

