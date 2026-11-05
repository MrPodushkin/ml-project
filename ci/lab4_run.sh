#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
python -m ci.lab4_secrets
name=ml-project-lab4
if sudo docker container inspect "$name" >/dev/null 2>&1; then
  test "$(sudo docker inspect -f '{{index .Config.Labels "ml-project.lab"}}' "$name")" = lab4
  sudo docker rm -f "$name"
fi
sudo docker run --detach --name "$name" --label ml-project.lab=lab4 \
  --restart unless-stopped --env-file "$HOME/.config/ml-project/lab4.env" \
  --publish 127.0.0.1:8000:8000 --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 64 --memory 768m --cpus 1 ml-project:lab4
