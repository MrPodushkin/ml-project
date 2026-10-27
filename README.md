
# SMS Spam Classification

## Описание проекта

Проект по классификации SMS-сообщений
на спам (spam) и обычные сообщения (ham).

Используется датасет SMS Spam Collection
из UCI Machine Learning Repository.

## Используемая модель

- TF-IDF
- Logistic Regression
- Scikit-learn

## Результаты

- Accuracy: 96.62%
- Precision (spam): 98.98%
- Recall (spam): 74.05%
- F1-score (spam): 84.72%

Размер тестовой выборки: 1034 сообщения.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Обучение модели

```bash
python src/train_model.py
```

## Оценка качества

```bash
python src/evaluate_model.py
```

## Автоматические тесты

```bash
python -m pytest tests/ -q
```

## Проверка устойчивости

```bash
python -m src.check_robustness
```

## Предсказание

```bash
python -m src.check_prediction
```

## Проверка модели после сохранения

```bash
python -m src.verify_model
```

## Безопасность

- Проверка входных SMS.
- Ограничение длины сообщений.
- Негативные тесты.
- Статический анализ Bandit.
- Контроль целостности модели с помощью SHA256.
- Безопасный запуск Jupyter через SSH-туннель.

## Источник данных

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

