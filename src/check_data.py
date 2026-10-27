
import pandas as pd

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("=== РАЗМЕР ДАТАСЕТА ===")
print(df.shape)

print("\n=== ТИПЫ ДАННЫХ ===")
print(df.dtypes)

print("\n=== ПРОПУСКИ ===")
print(df.isna().sum())

print("\n=== ПУСТЫЕ СООБЩЕНИЯ ===")
empty = df["message"].str.strip().eq("")
print(empty.sum())

print("\n=== ПОЛНЫЕ ДУБЛИКАТЫ ===")
print(df.duplicated().sum())

print("\n=== ДУБЛИКАТЫ ТЕКСТОВ ===")
print(df["message"].duplicated().sum())

print("\n=== ПРОВЕРКА МЕТОК ===")
print(df["label"].value_counts(dropna=False))

invalid = ~df["label"].isin(["ham", "spam"])
print("Некорректных меток:", invalid.sum())

print("\n=== РАСПРЕДЕЛЕНИЕ В ПРОЦЕНТАХ ===")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n=== ПРИМЕРЫ ДАННЫХ ===")
print(df.head())

