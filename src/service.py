"""Authenticated SMS API with streamed input limits and a supervised model worker."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from src.data_utils import ROOT
from src.model_utils import MODEL_HASH_PATH, MODEL_PATH
from src.predict_sms import InputValidationError, validate_sms
from src.service_metrics import Observability
from src.service_worker import ModelWorker

LOGGER = logging.getLogger("sms_api")
MODEL_VERSION = "sms-lab2-v1"
STATIC = ROOT / "static"


def read_secret(name):
    filename = os.environ.get(name + "_FILE")
    return (
        Path(filename).read_text(encoding="utf-8").strip() if filename else os.environ.get(name, "")
    )


@dataclass
class Settings:
    token: str = field(default_factory=lambda: read_secret("LAB4_API_TOKEN"), repr=False)
    metrics_token: str = field(
        default_factory=lambda: read_secret("LAB4_METRICS_TOKEN"), repr=False
    )
    model_path: Path = MODEL_PATH
    manifest: Path = MODEL_HASH_PATH
    max_body_bytes: int = 8192
    body_timeout: float = 2.0
    inference_timeout: float = 2.0
    startup_timeout: float = 15.0
    rate_limit: int = 20
    rate_period: float = 60.0
    detection_window: float = 10.0
    detection_threshold: int = 3

    def __post_init__(self):
        if len(self.token) < 32:
            raise ValueError("LAB4_API_TOKEN must contain at least 32 characters")
        if not self.metrics_token:
            self.metrics_token = self.token
        if len(self.metrics_token) < 32:
            raise ValueError("LAB4_METRICS_TOKEN must contain at least 32 characters")
        if not 1 <= self.rate_limit <= 1000 or not 0.1 <= self.rate_period <= 3600:
            raise ValueError("Invalid rate limit")
        if not 1 <= self.max_body_bytes <= 65536:
            raise ValueError("Invalid body limit")
        if not 0 < self.inference_timeout <= 10 or not 0 < self.body_timeout <= 10:
            raise ValueError("Invalid timeout")

    @classmethod
    def from_env(cls):
        return cls(
            rate_limit=int(os.environ.get("LAB4_RATE_LIMIT", "20")),
            rate_period=float(os.environ.get("LAB4_RATE_PERIOD", "60")),
            inference_timeout=float(os.environ.get("LAB4_INFERENCE_TIMEOUT", "2")),
            detection_window=float(os.environ.get("LAB4_DETECTION_WINDOW", "10")),
        )


class SMSInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: StrictStr


class RateLimiter:
    def __init__(self, limit, period):
        self.limit, self.period = limit, period
        self.clients = OrderedDict()

    def allow(self, key):
        now = time.monotonic()
        if key not in self.clients and len(self.clients) >= 1024:
            for old_key, history in list(self.clients.items()):
                if history[-1] <= now - self.period:
                    del self.clients[old_key]
            if len(self.clients) >= 1024:
                return False
        times = self.clients.setdefault(key, deque())
        self.clients.move_to_end(key)
        while times and times[0] <= now - self.period:
            times.popleft()
        if len(times) >= self.limit:
            return False
        times.append(now)
        # Expired client entries are evicted, keeping memory bounded.
        for old_key in list(self.clients)[:16]:
            if old_key != key and self.clients[old_key][-1] <= now - self.period:
                del self.clients[old_key]
        return True


def authenticated(request, secret):
    values = request.headers.getlist("authorization")
    if len(values) != 1 or len(values[0]) > 256:
        return False
    scheme, _, supplied = values[0].partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(
        hashlib.sha256(supplied.encode()).digest(), hashlib.sha256(secret.encode()).digest()
    )


def reject(request, status, reason, message):
    request.state.reason = reason
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else {}
    if status == 429:
        headers["Retry-After"] = str(max(1, int(request.app.state.settings.rate_period)))
    return JSONResponse(
        {"error": {"code": reason, "message": message}, "request_id": request.state.request_id},
        status,
        headers=headers,
    )


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON number")


def create_app(settings=None):
    settings = settings or Settings.from_env()
    worker = ModelWorker(
        settings.model_path, settings.manifest, settings.inference_timeout, settings.startup_timeout
    )
    monitoring = Observability(settings.detection_window, settings.detection_threshold)
    limiter = RateLimiter(settings.rate_limit, settings.rate_period)
    readers = asyncio.Semaphore(8)

    @asynccontextmanager
    async def lifespan(application):
        await asyncio.to_thread(worker.start)
        yield
        await worker.close()

    app = FastAPI(
        title="SMS Guard",
        version="4.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.worker = worker
    app.state.monitoring = monitoring
    app.state.settings = settings

    @app.middleware("http")
    async def observe(request, call_next):
        started = time.perf_counter()
        request.state.request_id = uuid.uuid4().hex
        request.state.reason = ""
        try:
            response = await call_next(request)
        except Exception:
            response = reject(request, 500, "internal_error", "Internal service error")
        route = (
            request.url.path
            if request.url.path
            in {
                "/",
                "/static/app.js",
                "/static/style.css",
                "/health",
                "/predict",
                "/metrics",
                "/monitoring/detection",
            }
            else "other"
        )
        method = request.method if request.method in {"GET", "POST", "HEAD", "OPTIONS"} else "OTHER"
        duration = time.perf_counter() - started
        monitoring.observe(route, method, response.status_code, duration, request.state.reason)
        LOGGER.info(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "request_id": request.state.request_id,
                    "route": route,
                    "method": method,
                    "status": response.status_code,
                    "duration_ms": round(duration * 1000, 3),
                    "model_version": MODEL_VERSION,
                    "rejection_type": request.state.reason or None,
                }
            )
        )
        response.headers.update(
            {
                "X-Request-ID": request.state.request_id,
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
                "style-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                "base-uri 'none'; form-action 'self'",
            }
        )
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/static/app.js")
    async def javascript():
        return FileResponse(STATIC / "app.js", media_type="application/javascript")

    @app.get("/static/style.css")
    async def stylesheet():
        return FileResponse(STATIC / "style.css", media_type="text/css")

    @app.get("/health")
    async def health():
        ready = worker.healthy
        return JSONResponse(
            {"status": "healthy" if ready else "unavailable", "model_version": MODEL_VERSION},
            200 if ready else 503,
        )

    @app.get("/metrics")
    async def metrics(request: Request):
        if not authenticated(request, settings.metrics_token):
            return reject(request, 401, "unauthorized", "Valid bearer token required")
        return Response(monitoring.render(worker.healthy), media_type="text/plain; version=0.0.4")

    @app.get("/monitoring/detection")
    async def detection(request: Request):
        if not authenticated(request, settings.metrics_token):
            return reject(request, 401, "unauthorized", "Valid bearer token required")
        return monitoring.detection()

    async def read_body(request):
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > settings.max_body_bytes:
                raise OverflowError("body_too_large")
            data.extend(chunk)
        return bytes(data)

    @app.post("/predict")
    async def predict(request: Request):
        address = request.client.host if request.client else "unknown"
        if not limiter.allow(address):
            return reject(request, 429, "rate_limit", "Request limit exceeded; retry later")
        if not authenticated(request, settings.token):
            return reject(request, 401, "unauthorized", "Valid bearer token required")
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            return reject(request, 415, "unsupported_media_type", "Use application/json")
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            return reject(
                request, 415, "unsupported_encoding", "Compressed request bodies are disabled"
            )
        if readers.locked():
            return reject(request, 503, "body_readers_busy", "Service is busy; retry later")
        async with readers:
            try:
                body = await asyncio.wait_for(read_body(request), timeout=settings.body_timeout)
            except OverflowError:
                return reject(request, 413, "body_too_large", "Request body exceeds 8192 bytes")
            except TimeoutError:
                return reject(request, 408, "body_timeout", "Request body read deadline exceeded")
        try:
            payload = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
            parsed = SMSInput.model_validate(payload)
            message = validate_sms(parsed.message)
        except ValueError, ValidationError, InputValidationError, RecursionError:
            return reject(
                request,
                422,
                "invalid_input",
                "Expected one nonempty UTF-8 message, "
                "at most 1000 characters; no extra fields, NUL, NaN or Inf",
            )
        try:
            before = worker.calls
            result = await worker.predict(message)
        except BlockingIOError:
            return reject(request, 503, "inference_busy", "One prediction is already running")
        except ConnectionError:
            return reject(request, 503, "model_unavailable", "Model is not ready")
        except TimeoutError:
            return reject(request, 504, "prediction_timeout", "Prediction deadline exceeded")
        except RuntimeError:
            return reject(request, 500, "prediction_failed", "Prediction failed")
        finally:
            if worker.calls > before:
                monitoring.predictions.inc()
        return {
            "prediction": result,
            "model_version": MODEL_VERSION,
            "request_id": request.state.request_id,
        }

    return app


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        "src.service:create_app",
        factory=True,
        host=os.environ.get("LAB4_HOST", "127.0.0.1"),
        port=int(os.environ.get("LAB4_PORT", "8000")),
        workers=1,
        access_log=False,
        proxy_headers=False,
        timeout_keep_alive=5,
        limit_concurrency=32,
        limit_max_requests=10000,
    )


if __name__ == "__main__":
    main()
