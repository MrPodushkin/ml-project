# Лабораторная работа №2 — SMS Spam Collection

Решение построено для проекта `~/ml-project`, созданного в ЛР1.

## Запуск

```bash
cd ~/ml-project
source .venv/bin/activate
python -m pip install pandas numpy scikit-learn matplotlib joblib pytest bandit jupyterlab nbformat
python -m src.train
python -m pytest tests/ -q
python -m src.robustness
python -m src.demo_predict
python -m bandit -r src/
python -m pip freeze > requirements.txt
sha256sum models/spam_pipeline.joblib
```

Исходный файл: `data/SMSSpamCollection` из ЛР1.

Не добавляйте в публичный Git `metrics/holdout_private.csv`, `metrics/errors_private.csv` и модель joblib. Из-за опасности pickle загружайте только собственный проверенный артефакт. Используйте приватное хранилище или ссылку для преподавателя. Метрики `metrics/baseline.json`, `metrics/data_check.json`, `metrics/robustness.json` и `metrics/model.sha256` можно версионировать после проверки содержимого.

В `metrics/baseline.json` записаны F1 для spam, accuracy, recall, precision, сравнение с DummyClassifier, параметры обучения, размеры выборок, SHA256 данных и матрица ошибок. В `metrics/robustness.json` — эксперимент на 20 сообщениях и нетипичный вход.

Датасет англоязычный; близкие тексты, связанные кампании и происхождение сообщений могут пересекаться между обучением и тестом. Удаление нормализованных дублей снижает риск дословной утечки, но не гарантирует независимость кампаний/источников: в исходных данных отсутствует надёжный групповой идентификатор. Ограничение описать в отчёте.

`MAX_SMS_CHARS = 1000` ограничивает число символов строки **после её получения**. Это не заменяет ограничение размера HTTP-запроса до декодирования в будущем API (ЛР4).
