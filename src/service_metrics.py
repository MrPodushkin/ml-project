"""Bounded-cardinality Prometheus metrics and an observable detection rule."""

import time
from collections import deque
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Observability:
    def __init__(self, window=10.0, threshold=3):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "sms_api_requests_total",
            "HTTP requests",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "sms_api_request_duration_seconds",
            "Response duration",
            ["route"],
            buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 15),
            registry=self.registry,
        )
        self.rejections = Counter(
            "sms_api_rejections_total",
            "Controlled rejections",
            ["reason"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "sms_api_predictions_total",
            "Requests sent to the real model",
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "sms_api_model_ready", "Model worker readiness", registry=self.registry
        )
        self.alert = Gauge(
            "sms_api_detection_active",
            "401/429 threshold in rolling window",
            registry=self.registry,
        )
        self.events = deque(maxlen=1024)
        self.window = window
        self.threshold = threshold

    def detection(self):
        now = time.monotonic()
        while self.events and self.events[0] <= now - self.window:
            self.events.popleft()
        active = len(self.events) >= self.threshold
        self.alert.set(int(active))
        return {
            "active": active,
            "events_in_window": len(self.events),
            "threshold": self.threshold,
            "window_seconds": self.window,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }

    def observe(self, route, method, status, duration, reason):
        self.requests.labels(route, method, str(status)).inc()
        self.duration.labels(route).observe(duration)
        if reason:
            self.rejections.labels(reason).inc()
        if route == "/predict" and status in (401, 429):
            self.events.append(time.monotonic())
        self.detection()

    def render(self, ready):
        self.model_ready.set(int(ready))
        self.detection()
        return generate_latest(self.registry)
