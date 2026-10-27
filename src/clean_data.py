
import pandas as pd

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

original_count = len(df)

df = df.dropna(subset=["label", "message"]).copy()
df = df[df["label"].isin(["ham", "spam"])].copy()
df = df[df["message"].str.strip().ne("")].copy()

# Проверяем конфликты: одинаковый текст с разными метками.
conflicts = df.groupby("message")["label"].nunique()
conflict_texts = conflicts[conflicts > 1].index

if len(conflict_texts):
    print("Есть конфликтующие метки. Нужна ручная проверка.")
    print(df[df["message"].isin(conflict_texts)])
    raise SystemExit("Очистка остановлена")

df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

df.to_csv("data/sms_clean.csv", index=False)

print("Исходных сообщений:", original_count)
print("После очистки:", len(df))
print("Удалено:", original_count - len(df))
print("Распределение классов:")
print(df["label"].value_counts())
