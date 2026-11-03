# ЛР2: фактические результаты после исправлений

Проверено на сервере 3 ноября 2026 года. Единственная модель: `models/spam_pipeline.joblib`. Jupyter Server и системные настройки не менялись. Ноутбуки выполнены через временные Python-ядра без запуска веб-сервера.

## Данные и обучение

5572 исходных SMS; удалено 415 нормализованных дублей, пропусков и конфликтующих меток нет. После очистки 5157 сообщений: ham 4515, spam 642. Seed 42, стратифицированное разделение train/test 4125/1032, нормализованное пересечение 0. TF-IDF обучается только на train. Параметры, версии и время обучения (0.2082 с в этом запуске) находятся в `metrics/baseline.json`.

Источник: UCI SMS Spam Collection. SHA256 исходного файла совпадает с эталоном ЛР1; изменения данных не производились. Существующих меток достаточно для учебного baseline обоих классов; ограниченная эпоха, английский язык, малое число spam и отсутствие идентификаторов кампаний не позволяют переносить качество на произвольный поток сообщений. Корректность разделения по нормализованному тексту не гарантирует независимости похожих кампаний.

| Метрика | Настоящая модель | DummyClassifier (ham) |
|---|---:|---:|
| accuracy | 0.978682 | 0.875969 |
| precision_spam | 0.895522 | 0.000000 |
| recall_spam | 0.937500 | 0.000000 |
| f1_spam | 0.916031 | 0.000000 |

Метрики обоих классов и способы усреднения сохранены в baseline. Матрица ошибок (строки — истина, столбцы — предсказание, порядок ham/spam): `[[890,14],[8,120]]`. Визуализация — `metrics/confusion_matrix.png` и выполненный notebook. 14 FP и 8 FN дают 22 ошибки из 1032.

Первые разобранные ошибки в ноутбуке включают короткие обычные сообщения о звонках, ошибочно помеченные spam, и сообщение с меткой spam, пропущенное как ham. Общая лексика звонков встречается в обоих классах; линейная модель не различает намерение и контекст. В ноутбуке опубликованы только ID, метки и размеры пяти ошибок; исходные тексты доступны владельцу локально в `metrics/errors_private.csv`. Неоднозначность меток — ограничение набора, а не основание менять test после оценки.

## Пункт 6: настоящие тесты

34 теста прошли. Все модельные тесты используют собственный сохранённый Pipeline; mock, заглушек и подмены predict нет. `sys.setprofile` только наблюдает вызовы реального Pipeline: негативный вход не достигает модели, положительный вызывает её один раз. Существующая модель не подменяется ради теста ошибки. Ниже каждый параметризованный случай из свежего JUnit XML.

| Тест / входной случай | Ожидаемое поведение | Фактический результат | Вывод |
|---|---|---|---|
| `test_real_dataset_integrity_and_no_normalized_overlap` | SHA256 ЛР1 совпадает; train/test 4125/1032; пересечение 0 | Все соответствующие assertions выполнены | PASS |
| `test_loaded_real_model_reproduces_baseline` | Предсказания загруженной модели воспроизводят F1 baseline; SHA256 совпадает | Все соответствующие assertions выполнены | PASS |
| `test_robustness_uses_real_model_test_only` | 20 уникальных ID принадлежат test; пересечение train 0; модель и split совпадают | Все соответствующие assertions выполнены | PASS |
| `test_small_typo_changes_only_one_character` | Изменён ровно один символ при сохранении длины | Все соответствующие assertions выполнены | PASS |
| `test_compat_prediction_matches_actual_pipeline` | Совпадение совместимого API с прямым predict настоящей модели | Все соответствующие assertions выполнены | PASS |
| `test_compat_invalid_message` | InputValidationError для None | Все соответствующие assertions выполнены | PASS |
| `test_compat_length_limit` | InputValidationError для 21 символа при лимите 20 | Все соответствующие assertions выполнены | PASS |
| `test_compat_valid_message` | Корректная строка возвращается без изменения | Все соответствующие assertions выполнены | PASS |
| `test_invalid_limit[0]` | ValueError для недопустимого значения лимита | Все соответствующие assertions выполнены | PASS |
| `test_invalid_limit[-1]` | ValueError для недопустимого значения лимита | Все соответствующие assertions выполнены | PASS |
| `test_invalid_limit[True]` | ValueError для недопустимого значения лимита | Все соответствующие assertions выполнены | PASS |
| `test_invalid_limit[1.5]` | ValueError для недопустимого значения лимита | Все соответствующие assertions выполнены | PASS |
| `test_invalid_limit[1001]` | ValueError для недопустимого значения лимита | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[none]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[integer]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[list]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[dict]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[empty]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[whitespace]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[nan]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[inf]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[oversized]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[nul]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_invalid_inputs_never_reach_real_model[invalid_unicode]` | InputValidationError; 0 вызовов настоящего Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_valid_inputs_call_real_model_once[Hello, how are you?]` | Класс ham/spam; 1 вызов настоящего Pipeline; совпадение с прямым predict | Все соответствующие assertions выполнены | PASS |
| `test_valid_inputs_call_real_model_once[See you tomorrow at 5 PM.]` | Класс ham/spam; 1 вызов настоящего Pipeline; совпадение с прямым predict | Все соответствующие assertions выполнены | PASS |
| `test_valid_inputs_call_real_model_once[Congratulations! Claim your free prize now!]` | Класс ham/spam; 1 вызов настоящего Pipeline; совпадение с прямым predict | Все соответствующие assertions выполнены | PASS |
| `test_valid_inputs_call_real_model_once[\U0001f381 Special offer!]` | Класс ham/spam; 1 вызов настоящего Pipeline; совпадение с прямым predict | Все соответствующие assertions выполнены | PASS |
| `test_valid_inputs_call_real_model_once[\u041f\u0440\u0438\u0432\u0435\u0442! \u041a\u0430\u043a \u0434\u0435\u043b\u0430?]` | Класс ham/spam; 1 вызов настоящего Pipeline; совпадение с прямым predict | Все соответствующие assertions выполнены | PASS |
| `test_deterministic_real_prediction` | Одинаковый реальный ответ при двух вызовах | Все соответствующие assertions выполнены | PASS |
| `test_real_model_boundary` | 1000 символов приняты; 1001 отклонён до вызова Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_small_artificial_limit_with_real_model` | 20 символов приняты; 21 отклонён до вызова Pipeline | Все соответствующие assertions выполнены | PASS |
| `test_missing_model` | FileNotFoundError для отсутствующего файла | Все соответствующие assertions выполнены | PASS |
| `test_corrupt_real_artifact_rejected_before_deserialization` | ValueError по SHA256 до десериализации повреждённого файла | Все соответствующие assertions выполнены | PASS |

Полные журналы: `metrics/pytest_results.txt`, `metrics/pytest_results.xml`. Искусственный предел 20/21 позволяет проверить превышение без большого входа. Рабочий предел 1000 покрывает составные SMS и ограничивает обработку; это не доказательство принадлежности распределению и не ограничение размера HTTP-запроса до его декодирования.

## Пункт 7: парный эксперимент

10 spam и 10 ham заранее выбираются из test с seed 42. ID объектов сохранены в `metrics/robustness.json`. Выбор не зависит от успешности предсказаний. Отпечатки данных, модели и split проверены; пересечение экспериментальных SMS с train — 0. Оригинал и изменённая версия имеют одну истинную метку. Парные тексты и предсказания сохранены локально в игнорируемом `metrics/robustness_private.csv`.

| Изменение | Изменившихся ответов | Доля | F1 spam до | F1 spam после | Accuracy после |
|---|---:|---:|---:|---:|---:|
| Регистр и крайние пробелы | 0 / 20 | 0% | 0.9474 | 0.9474 | 0.9500 |
| Одна символьная опечатка | 1 / 20 | 5% | 0.9474 | 1.0000 | 1.0000 |
| Дополнительный стресс: массовая замена букв цифрами | 5 / 20 | 25% | 0.9474 | 0.5714 | 0.7000 |

Отсутствие эффекта регистра/пробелов ожидаемо: TF-IDF приводит регистр и разбивает текст на токены. Одна опечатка изменила один ответ и случайно исправила одну исходную ошибку; это показывает чувствительность, а не доказанное улучшение качества от опечаток. Массовые замены — отдельное более сильное воздействие; они изменили 5 ответов и снизили F1. Эксперимент на 20 объектах не доказывает устойчивость ко всем атакам.

Нетипичный синтетический вход из emoji принят по схеме, но имеет 0 ненулевых признаков TF-IDF и получает ham. Это реакция модели при отсутствии словарных признаков, а не подтверждение её пригодности для такого текста. Без обоснования распределения вход не объявляется OOD. Превышение 1000 символов отдельно отклонено контролируемой ошибкой.

Предложенная мера: оценить нормализацию типичных замен на отдельной validation/CV внутри train, не подбирая защиту по test. Для нулевого словарного покрытия рассмотреть отказ или ручную обработку. Эти дополнительные политики пока не реализованы.

## Пункты 8–10: артефакты и ограничения

Старый бинарный артефакт и локальные несогласованные отчёты перемещены в приватный резервный каталог владельца. В проекте одна модель, один корневой manifest `model.sha256`, baseline, evaluation и robustness с одинаковой привязкой к модели/данным. Модель успешно загружена и дала предсказания. requirements.txt соответствует установленным версиям; pip check прошёл.

Архив датасета исключён из текущего Git, локально сохранён и игнорируется. Старые коммиты не переписывались. Ноутбуки не содержат исходных SMS в выводах. Jupyter Server оставлен без изменений по указанию владельца; новый факт его защищённого запуска не заявляется.

Свежий Bandit: 0 замечаний, без nosec и исключённых проверок. Обоснование первоначального B311 и устранение случайного дублирующего скрипта — в `docs/security.md`.

```text
f054edc2c35248ad9cb26a78ce7d8be044558fe39d3d73d8cf487777bbfeb06a  models/spam_pipeline.joblib
```
