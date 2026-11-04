#!/usr/bin/env bash
# Explicit opt-in; the default application paths continue to select the Lab 2 artifact.
export ML_MODEL_PATH=ci-artifacts/runtime/spam_pipeline.joblib
export ML_MODEL_MANIFEST=ci-artifacts/runtime/model.sha256
export ML_EXPERIMENT_METADATA=ci-artifacts/experiment.json
export ML_ROBUSTNESS_REPORT=ci-artifacts/robustness.json
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
