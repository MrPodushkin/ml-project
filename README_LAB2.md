# Лабораторная работа №2

Актуальная инструкция: [README.md](README.md). Таблица фактических результатов и тестов: [docs/lab2-results.md](docs/lab2-results.md).

Основные команды из корня проекта:

```bash
source .venv/bin/activate
python -m src.train
python -m src.evaluate_model
python -m src.robustness
python -m pytest tests/ -q
python -m src.verify_model
python -m bandit -r src/
```

Используется только `models/spam_pipeline.joblib`, контрольная сумма в `model.sha256`. Эксперимент выбирает 10 spam и 10 ham из test этой же модели. Отпечатки данных, разделения и модели проверяются перед экспериментом. Ни одна из двадцати SMS не входит в train.

Тесты работают с настоящей моделью без mock. Невызов Pipeline на некорректном входе проверяется наблюдением реальных вызовов, без подмены методов.
