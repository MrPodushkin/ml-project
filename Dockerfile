FROM python:3.14.10-slim-trixie@sha256:930e1abba7c4e8f96ed93e8012c35d174d7732752731bbd5e8e757d7ec6f3966 AS python_runtime
FROM ubuntu:26.04@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    LAB4_HOST=0.0.0.0
ENV PATH="/usr/local/bin:$PATH"
WORKDIR /app

# A patched glibc runtime supports the pinned manylinux wheels without a compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates libbz2-1.0 libexpat1 libffi8 libgomp1 liblzma5 \
       libssl3t64 libstdc++6 passwd zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=python_runtime /usr/local /usr/local
RUN ldconfig

COPY requirements-runtime.txt requirements-service.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-service.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY src/__init__.py src/data_utils.py src/model_utils.py src/predict_sms.py src/container_predict.py ./src/
COPY src/service.py src/service_worker.py src/service_metrics.py src/service_health.py src/service_smoke.py ./src/
COPY static/ ./static/
COPY ci-artifacts/runtime/spam_pipeline.joblib ./models/spam_pipeline.joblib
COPY ci-artifacts/runtime/model.sha256 ./model.sha256
RUN sha256sum --check model.sha256

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "src.service_health"]
ENTRYPOINT ["python", "-m", "src.service"]
