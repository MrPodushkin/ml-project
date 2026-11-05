# ЛР4. Защищённый API, веб-интерфейс и мониторинг SMS-модели

Сервис использует сохранённый Pipeline ЛР2: TF-IDF + LogisticRegression. Новая модель
не обучается для веб-сервиса. Версия API-ответа: `sms-lab2-v1`.
Эталон SHA256 находится в отдельном `model.sha256`; исходный артефакт:
`f054edc2c35248ad9cb26a78ce7d8be044558fe39d3d73d8cf487777bbfeb06a`.
Хэш проверяется перед `joblib.load`. Пользователь не может загружать модели или менять путь.
Хэш обнаруживает изменение файла; доверие к источнику pickle остаётся обязательным.

## Запуск на учебном сервере

```bash
cd ~/ml-project
python3 -m venv .lab4-venv
source .lab4-venv/bin/activate
python -m pip install --only-binary=:all: -r requirements-ci.txt
source ci/env.sh
python -m ci.prepare
sudo docker build --pull -t ml-project:lab4 .
bash ci/lab4_run.sh
```

`ci/lab4_run.sh` создаёт секреты только при их отсутствии в
`~/.config/ml-project/lab4.env` (права 600, каталог 700), передаёт их через
`docker --env-file`, публикует **127.0.0.1:8000**. Секретов нет в Git, HTML или образе.
Docker администраторы могут читать runtime environment; это ограничение учебного размещения.
Повторный запуск заменяет только контейнер с именем и меткой этой лабораторной работы.
Образ ЛР3 `ml-project:lab3` сохраняется; новый образ имеет тег `ml-project:lab4`.

Без Docker, из того же окружения после подготовки артефакта:

```bash
python -m ci.lab4_secrets
set -a
source ~/.config/ml-project/lab4.env
set +a
python -m src.service
```

На Windows открыть туннель в отдельном PowerShell:

```powershell
ssh -N -L 8000:127.0.0.1:8000 mluser@192.168.50.109
```

Открыть `http://127.0.0.1:8000`. Для заполнения поля токена посмотреть строку
`LAB4_API_TOKEN` в приватном env-файле на сервере. Не включать её в скриншоты.
В форме поле скрыто; значение хранится только в памяти страницы, не в localStorage,
не в URL и не в логах. Предусмотрены ожидание, результат, ошибки 401/413/415/422/429/503/504.
Недоверенные значения выводятся через `textContent`. Страница и API одного origin; CORS не включён.
Передача токена допустима через локальный loopback или SSH-туннель; открытый HTTP через сеть не используется.

## Контракт API

`POST /predict`, `Content-Type: application/json`, `Authorization: Bearer <token>`:

```json
{"message": "Hi! Are we meeting at six?"}
```

```json
{"prediction": "ham", "model_version": "sms-lab2-v1", "request_id": "generated-per-request"}
```

Единственное обязательное поле — `message`: строгая строка из 1–1000 символов после
проверки на пустоту. Пустые/пробельные строки, NUL, некорректный Unicode, дополнительные
поля, дублирующиеся JSON-ключи, NaN/Inf, неверные типы и повреждённый JSON отклоняются.
Предобработка и `predict_sms` совпадают с ЛР2. Возвращается класс, без выдуманной вероятности.
Качество на русском языке не подтверждено: исходный датасет англоязычный.

`GET /health`: 200 `healthy` только при живом процессе с успешно проверенной и загруженной
моделью; иначе 503 `unavailable`. В ответе нет путей, traceback или секретов.
Ошибка SHA256, отсутствие артефакта и ошибка десериализации проверены отдельно.

`GET /metrics` и `GET /monitoring/detection`: отдельный bearer-токен
`LAB4_METRICS_TOKEN` (в тестах допускается использование API-токена по умолчанию).
Prometheus может использовать `authorization.credentials_file` с этим токеном.
Контрольное приложение не доверяет `X-Forwarded-For` (`proxy_headers=False`).

## Пределы ресурсов и HTTP-коды

| Ограничение | Значение и обоснование |
|---|---|
| Тело запроса | 8192 байт: хватает JSON для 1000 Unicode-символов; считается по фактическим ASGI-чанкам до накопления всего тела. Content-Length не является доказательством размера. |
| Длина SMS | До 1000 Unicode-символов, как в ЛР2. Batch, файлы и сжатые тела не принимаются. |
| Чтение тела | До 2 секунд, максимум 8 одновременных читателей; нет бесконечной очереди. |
| Предсказание | Один отдельный процесс модели; максимум одно предсказание одновременно, без очереди. |
| Deadline | 2 секунды ожидания результата IPC. При превышении процесс получает terminate, затем при необходимости kill, и обязательно join. Новый процесс загружает модель заново; до готовности health=503. |
| Восстановление | Загрузка нового worker ограничена 15 секундами; при неуспехе сервис остаётся неготовым. Это не повторная загрузка при каждом запросе. |
| Rate limit | 20 запросов `/predict` за скользящие 60 секунд по фактическому IP клиента, включая неверные токены и невалидные запросы. |
| Сервер | Один Uvicorn worker, до 32 соединений/задач по лимиту сервера, keepalive 5 секунд. |
| Контейнер | UID/GID 10001, read-only, cap-drop ALL, no-new-privileges, 64 PID, 768 MiB, 1 CPU. |

Коды: 401 — нет/неверен токен; 413 — превышено тело; 415 — неверный Content-Type
или сжатое тело; 422 — JSON/схема/содержимое SMS; 408 — deadline чтения;
429 — rate limit; 503 — модель недоступна или занята; 504 — deadline предсказания;
500 — контролируемая внутренняя ошибка. Приоритет: limiter → аутентификация → формат → поток → схема → модель.
Ошибка содержит безопасные `error.code`, `error.message`, `request_id`; traceback клиенту не выдаётся.

Limiter и правило обнаружения находятся в памяти одного процесса, сбрасываются при
перезапуске и не согласованы между несколькими Uvicorn workers. Адреса ограничены
1024 записями; при заполнении действует отказ. Через SSH все пользователи могут
иметь общий адрес loopback. Для нескольких workers требуется общее хранилище/шлюз.

## Тестирование и короткие команды

```bash
python -m pytest tests/ -q
python -m pytest tests/test_api.py -q
ruff check src/ tests/ ci/
python -m bandit -r src/
python -m pip_audit -r requirements-service.txt --no-deps --disable-pip
python -m ci.lab4_client health
python -m ci.lab4_client predict
python -m ci.lab4_client predict --message "FREE cash prize! Claim now!!!"
python -m ci.lab4_client unauthorized
python -m ci.lab4_client invalid
python -m ci.lab4_client oversized
python -m ci.lab4_client metrics
python -m ci.lab4_client detection
```

Клиент читает приватный env-файл самостоятельно и не печатает токены.
Тесты используют сохранённую модель, настоящий ASGI/HTTP-клиент и отдельный процесс.
Mocks, monkeypatch и подмена `predict` отсутствуют. Счётчик IPC-вызовов подтверждает,
что отклонённые запросы не переданы worker. Для hard timeout на Linux используется
SIGSTOP только собственного дочернего процесса теста; затем проверяются завершение,
освобождение слота и успешное предсказание новым процессом.

## Логи, метрики и обнаружение

JSON-логи: UTC-время, сгенерированный request_id, фиксированный маршрут, метод,
HTTP-статус, длительность, версия модели, тип отказа. Ввод, токены, IP и query string
не записываются. Отсутствуют labels с request_id, текстом и произвольными путями.

Метрики: `sms_api_requests_total{route,method,status}` (в том числе 4xx/5xx),
`sms_api_request_duration_seconds` (histogram), `sms_api_rejections_total{reason}`,
`sms_api_predictions_total`, `sms_api_model_ready`, `sms_api_detection_active`.
Учитываются отказы, дошедшие до приложения. Отказы Uvicorn на HTTP-парсинге и его
concurrency limit находятся на уровне сервера и в счётчики приложения не входят.
Reverse proxy не установлен; мониторинг отказов proxy здесь не заявляется.

Рабочее правило: не менее 3 ответов 401/429 на `/predict` за последние 10 секунд.
Проверка защищённого `/monitoring/detection` или `/metrics` пересчитывает скользящее
окно и показывает восстановление. Это локальное правило с gauge, не внешний Alertmanager.

Воспроизводимый безопасный сценарий:

```bash
python -m ci.lab4_demo
python -m ci.lab4_evidence result
python -m ci.lab4_evidence logs
python -m ci.lab4_evidence before
python -m ci.lab4_evidence after
python -m ci.lab4_evidence recovered
```

Запускается отдельный временный сервис на loopback, с новым секретом в памяти,
лимитом 4 запроса / 2 секунды и правилом 3 события / 3 секунды. Один нормальный
запрос даёт 200; три неверных токена дают 401, следующий запрос — 429. Правило
активируется; после ожидания 3.2 секунды нормальный запрос даёт 200, правило
возвращается в `active=false`. Результат содержит фактический UTC-интервал.
Это ограниченная проверка реакции сервиса, не доказательство атаки на модель
или ухудшения её точности. Prometheus/Grafana необязательны и в эту реализацию не включены.

## Контейнер и CI

Dockerfile использует закреплённые официальные базовые образы ЛР3, добавляет API,
static/ и только runtime-зависимости. Модель проверяется при сборке и запуске.
Загрузка выполняется один раз на поколение worker. HEALTHCHECK делает настоящий
HTTP-запрос к `/health`. Runtime-секреты создаются и передаются при запуске.

```bash
LAB3_DOCKER_SUDO=1 bash ci/run_local.sh
python -m ci.container_api_smoke --sudo
```

Сохраняются Ruff, Bandit, pytest, аудит полного дерева runtime-зависимостей API,
data/model gates, Trivy Dockerfile и реального собранного образа, а также правило
блокировки CRITICAL/UNKNOWN из ЛР3. HIGH не скрываются и не объявляются отсутствующими.
Добавлены API-тесты, сценарий обнаружения/восстановления и настоящий HTTP-smoke контейнера.
Автоматические действия закреплены SHA, permissions=contents:read, timeout=25 минут;
отчёты загружаются `if: always()` без превращения проваленного gate в успех.
