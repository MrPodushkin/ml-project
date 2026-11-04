#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p ci-artifacts
python -m ci.record_stage initialize not_run
current_stage=""
finish() {
  rc=$?
  trap - EXIT
  if [[ $rc -ne 0 && -n "$current_stage" ]]; then
    python -m ci.record_stage "$current_stage" failed
  fi
  printf '\nPipeline exit code: %s\n' "$rc"
  exit "$rc"
}
trap finish EXIT
run() {
  current_stage="$1"
  shift
  printf '\n=== STAGE %s ===\n' "$current_stage"
  python -m ci.record_stage "$current_stage" running
  "$@"
  python -m ci.record_stage "$current_stage" passed
}
docker_command=(docker)
if [[ "${LAB3_DOCKER_SUDO:-0}" == 1 ]]; then
  sudo -S -p '' -v
  docker_command=(sudo -n docker)
fi
verify_nonroot() {
  "${docker_command[@]}" run --rm --entrypoint id ml-project:lab3 -u > ci-artifacts/container-uid.txt
  cat ci-artifacts/container-uid.txt
  test "$(cat ci-artifacts/container-uid.txt)" != 0
}
smoke_check() {
  "${docker_command[@]}" run --rm --read-only --network none --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --memory 512m "$image" --smoke | tee ci-artifacts/container-smoke.json
}
image=ml-project:lab3
run artifacts python -m ci.prepare
python -m ci.metadata
run lint bash -o pipefail -c 'ruff check src/ tests/ ci/ | tee ci-artifacts/ruff.txt'
run bandit python -m bandit -r src/ -f json -o ci-artifacts/bandit.json
run tests bash -o pipefail -c 'python -m pytest tests/ -q --junitxml=ci-artifacts/pytest.xml | tee ci-artifacts/pytest.txt'
run dependency_audit python -m pip_audit -r requirements-runtime.txt --no-deps --disable-pip --format json --output ci-artifacts/pip-audit.json
run security_demonstration python -m ci.security_gate --demo --output ci-artifacts/security-demonstration.json
python -m ci.demonstrate
run data_gate python -m ci.data_gate
run evaluate python -m ci.evaluate
run model_gate python -m ci.model_gate
run install_scanner python -m ci.install_trivy
trivy="$HOME/.local/bin/trivy"
"$trivy" --version > ci-artifacts/trivy-version.txt
run docker_build "${docker_command[@]}" build --pull -t "$image" .
run config_scan "$trivy" config --severity HIGH,CRITICAL,UNKNOWN --exit-code 1 --format json --output ci-artifacts/trivy-config.json Dockerfile
run image_export "${docker_command[@]}" save --output ci-artifacts/image.tar "$image"
run image_scan "$trivy" image --input ci-artifacts/image.tar --scanners vuln --format json --output ci-artifacts/trivy-image.json
"$trivy" --version > ci-artifacts/trivy-version.txt
run security_gate python -m ci.security_gate
run non_root verify_nonroot
run smoke smoke_check
"${docker_command[@]}" image inspect "$image" --format '{"image_id":"{{.Id}}","user":"{{.Config.User}}"}' > ci-artifacts/container-inspect.json
printf '\nAll required stages PASSED\n'
