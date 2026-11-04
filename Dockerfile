FROM python:3.14.4-slim-trixie@sha256:2ca02f32b4d9d893863367ce07ec1972819f476dd38d8612f2a9cb6a41cbb727

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
WORKDIR /app

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
