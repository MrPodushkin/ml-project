import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(
    "data/SMSSpamCollection", sep="\t", header=None, names=["label", "message"], encoding="utf-8"
)

print("Размер датасета:", df.shape)
print("\nРаспределение классов:")
print(df["label"].value_counts())
print("\nПроцентное соотношение:")
print(df["label"].value_counts(normalize=True) * 100)

counts = df["label"].value_counts()

counts.plot(kind="bar")
plt.title("Распределение SMS по классам")
plt.xlabel("Класс")
plt.ylabel("Количество сообщений")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("data/class_distribution.png")
plt.show()
