FROM python:3.14.10-slim-trixie@sha256:930e1abba7c4e8f96ed93e8012c35d174d7732752731bbd5e8e757d7ec6f3966

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
WORKDIR /app

# Apply available fixes; SQLite is unused by inference and has unresolved findings.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade ca-certificates liblzma5 perl-base \
    && apt-get purge -y --auto-remove libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-runtime.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY src/__init__.py src/data_utils.py src/model_utils.py src/predict_sms.py src/container_predict.py ./src/
COPY ci-artifacts/runtime/spam_pipeline.joblib ./models/spam_pipeline.joblib
COPY ci-artifacts/runtime/model.sha256 ./model.sha256
RUN sha256sum --check model.sha256

USER 10001:10001
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-m", "src.container_predict", "--smoke"]
ENTRYPOINT ["python", "-m", "src.container_predict"]
CMD ["--smoke"]
