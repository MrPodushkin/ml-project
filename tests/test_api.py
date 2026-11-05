"""Real persisted model, real ASGI client and real process failures; no mocks."""

import asyncio
import json
import logging
import os
import signal
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from src.model_utils import MODEL_HASH_PATH, MODEL_PATH
from src.service import RateLimiter, Settings, create_app

TOKEN = "test-only-api-token-not-a-runtime-secret-123456"
AUTH = {"Authorization": "Bearer " + TOKEN}


@pytest.fixture(scope="module")
def service():
    app = create_app(Settings(token=TOKEN, rate_limit=1000))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        yield app, client


def test_healthy_page_and_real_predictions(service):
    app, client = service
    pid = app.state.worker.process.pid
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    for text, expected in [
        ("Hi! Are we meeting at six?", "ham"),
        ("FREE cash prize! Claim now!!!", "spam"),
    ]:
        response = client.post("/predict", headers=AUTH, json={"message": text})
        assert response.status_code == 200
        assert response.json()["prediction"] == expected
        assert response.json()["model_version"] == "sms-lab2-v1"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert app.state.worker.process.pid == pid
    assert app.state.worker.generation == 1


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer incorrect"}])
def test_auth_rejection_without_inference(service, headers):
    app, client = service
    calls = app.state.worker.calls
    response = client.post("/predict", headers=headers, json={"message": "hello"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert app.state.worker.calls == calls


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "  "},
        {"message": None},
        {"message": 123},
        {"message": []},
        {"message": {}},
        {"message": True},
        {"message": "A" * 1001},
        {"message": "nul\x00byte"},
        {"message": "\ud800"},
        {"message": "hello", "model_path": "/tmp/upload.joblib"},
        [],
        "hello",
    ],
)
def test_invalid_schema_never_reaches_real_model(service, payload):
    app, client = service
    calls = app.state.worker.calls
    # ensure_ascii also lets the invalid surrogate reach the actual JSON decoder.
    response = client.post(
        "/predict",
        headers={**AUTH, "Content-Type": "application/json"},
        content=json.dumps(payload, ensure_ascii=True),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert app.state.worker.calls == calls


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"\xff",
        b'{"message":NaN}',
        b'{"message":Infinity}',
        b'{"message":-Infinity}',
        b'{"message":"a","message":"b"}',
        b"[" * 1500 + b"]" * 1500,
    ],
)
def test_invalid_raw_json(service, body):
    app, client = service
    calls = app.state.worker.calls
    response = client.post(
        "/predict", headers={**AUTH, "Content-Type": "application/json"}, content=body
    )
    assert response.status_code == 422
    assert app.state.worker.calls == calls


@pytest.mark.parametrize(
    "headers,body,status",
    [
        ({"Content-Type": "text/plain"}, "hello", 415),
        ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, "compressed", 415),
        ({"Content-Type": "application/json"}, "A" * 8193, 413),
    ],
)
def test_body_and_format_rejections(service, headers, body, status):
    app, client = service
    calls = app.state.worker.calls
    response = client.post("/predict", headers={**AUTH, **headers}, content=body)
    assert response.status_code == status
    assert app.state.worker.calls == calls


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_stream_limit_without_trusting_content_length(declared_length):
    async def exercise():
        app = create_app(Settings(token=TOKEN))
        async with app.router.lifespan_context(app):
            calls = app.state.worker.calls

            async def chunks():
                for _ in range(12):
                    yield b"A" * 1024

            headers = {**AUTH, "Content-Type": "application/json"}
            if declared_length is not None:
                headers["Content-Length"] = declared_length
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post("/predict", headers=headers, content=chunks())
            assert response.status_code == 413
            assert app.state.worker.calls == calls

    asyncio.run(exercise())


def test_body_read_deadline():
    async def exercise():
        app = create_app(Settings(token=TOKEN, body_timeout=0.05))
        async with app.router.lifespan_context(app):

            async def slow():
                yield b'{"message":'
                await asyncio.sleep(0.2)
                yield b'"hello"}'

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/predict", headers={**AUTH, "Content-Type": "application/json"}, content=slow()
                )
            assert response.status_code == 408
            assert app.state.worker.calls == 0

    asyncio.run(exercise())


def test_rate_limit_detection_and_recovery():
    app = create_app(
        Settings(
            token=TOKEN, rate_limit=3, rate_period=0.3, detection_window=0.3, detection_threshold=3
        )
    )
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/predict", json={"message": "hello"}).status_code == 401
        assert client.get("/monitoring/detection", headers=AUTH).json()["active"] is True
        response = client.post("/predict", headers=AUTH, json={"message": "hello"})
        assert response.status_code == 429
        assert app.state.worker.calls == 0
        time.sleep(0.35)
        assert client.post("/predict", headers=AUTH, json={"message": "hello"}).status_code == 200
        assert client.get("/monitoring/detection", headers=AUTH).json()["active"] is False


@pytest.mark.parametrize("failure", ["hash", "deserialization", "missing"])
def test_model_startup_failure_is_never_healthy(tmp_path, failure):
    model = tmp_path / "spam_pipeline.joblib"
    manifest = tmp_path / "model.sha256"
    if failure != "missing":
        model.write_bytes(b"invalid executable model artifact")
    if failure == "deserialization":
        from hashlib import sha256

        manifest.write_text(sha256(model.read_bytes()).hexdigest() + "  " + model.name)
    else:
        manifest.write_text(MODEL_HASH_PATH.read_text())
    app = create_app(Settings(token=TOKEN, model_path=model, manifest=manifest))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unavailable"
        assert str(tmp_path) not in response.text
        assert client.post("/predict", headers=AUTH, json={"message": "hello"}).status_code == 503
        assert app.state.worker.calls == 0


@pytest.mark.skipif(os.name != "posix", reason="real Linux process SIGSTOP exercise")
def test_hard_timeout_busy_slot_and_real_worker_recovery():
    async def exercise():
        app = create_app(Settings(token=TOKEN, inference_timeout=0.1))
        async with app.router.lifespan_context(app):
            old_pid = app.state.worker.process.pid
            os.kill(old_pid, signal.SIGSTOP)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                first = asyncio.create_task(
                    client.post("/predict", headers=AUTH, json={"message": "hello"})
                )
                while app.state.worker.calls == 0:
                    await asyncio.sleep(0.005)
                second = await client.post("/predict", headers=AUTH, json={"message": "hello"})
                assert second.status_code == 503
                assert second.json()["error"]["code"] == "inference_busy"
                assert (await first).status_code == 504
                with pytest.raises(ProcessLookupError):
                    os.kill(old_pid, 0)
                for _ in range(300):
                    if app.state.worker.healthy:
                        break
                    await asyncio.sleep(0.05)
                assert app.state.worker.healthy
                assert app.state.worker.process.pid != old_pid
                assert (
                    await client.post("/predict", headers=AUTH, json={"message": "hello"})
                ).status_code == 200

    asyncio.run(exercise())


def test_metrics_protected_and_no_sensitive_logging(service, caplog):
    app, client = service
    with caplog.at_level(logging.INFO, logger="sms_api"):
        secret_text = "PRIVATE-SMS-should-never-appear-in-logs"
        assert (
            client.post("/predict", headers=AUTH, json={"message": secret_text}).status_code == 200
        )
        assert client.get("/metrics").status_code == 401
        response = client.get("/metrics", headers=AUTH)
        assert response.status_code == 200
        assert "sms_api_requests_total" in response.text
        assert "sms_api_rejections_total" in response.text
        assert "sms_api_predictions_total" in response.text
        assert "request_id=" not in response.text
        assert TOKEN not in response.text
        events = [
            json.loads(record.message) for record in caplog.records if record.name == "sms_api"
        ]
        assert events
        for event in events:
            assert {
                "time",
                "request_id",
                "route",
                "status",
                "duration_ms",
                "model_version",
                "rejection_type",
            } <= event.keys()
        assert secret_text not in caplog.text
        assert TOKEN not in caplog.text


def test_separate_metrics_token():
    app = create_app(
        Settings(token=TOKEN, metrics_token="separate-metrics-token-not-runtime-secret-1234")
    )
    with TestClient(app) as client:
        assert client.get("/metrics", headers=AUTH).status_code == 401
        assert (
            client.get(
                "/metrics",
                headers={"Authorization": "Bearer separate-metrics-token-not-runtime-secret-1234"},
            ).status_code
            == 200
        )


def test_original_model_and_static_safety(service):
    app, client = service
    assert MODEL_PATH.is_file()
    js = client.get("/static/app.js").text
    assert "innerHTML" not in js
    assert "localStorage" not in js
    assert TOKEN not in js
    assert client.get("/").headers["Content-Security-Policy"]


def test_empty_server_secret_refused():
    with pytest.raises(ValueError, match="at least 32"):
        Settings(token="")


def test_limiter_capacity_recovers_after_real_window_expiry():
    limiter = RateLimiter(limit=1, period=0.1)
    for index in range(1024):
        assert limiter.allow(f"test-client-{index}")
    assert not limiter.allow("new-client")
    assert len(limiter.clients) <= 1024
    time.sleep(0.11)
    assert limiter.allow("new-client")
    assert len(limiter.clients) <= 1024
