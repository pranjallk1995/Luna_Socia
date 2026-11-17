#!/usr/bin/env bash
# Starts Jupyter beside the running Luna Socia: her code from this branch, her data read-only,
# her models over Sentia's network, and the GPU's counters (read only: nothing here loads onto the GPU
# by itself). Prints the address to open. ./start.sh stop stops it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
name=luna_socia-notebooks
if [ "${1:-}" = stop ]; then docker rm -f "$name" >/dev/null && echo "Stopped."; exit 0; fi
docker inspect luna_socia-ui-1 >/dev/null 2>&1 || { echo "Start Luna Socia first."; exit 1; }
umask 077
# Socia's settings as the running app has them (with secrets: kept out of git, readable by you only).
docker inspect luna_socia-ui-1 --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -vE '^(HOME|PATH|PYTHONPATH)=' > "$here/.socia.env"
[ -s "$here/.jupyter-token" ] || openssl rand -hex 24 > "$here/.jupyter-token"
umask 022
docker build -q -t luna-socia-notebooks:local "$here" >/dev/null
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --network luna_sentia_network --env-file "$here/.socia.env" \
  --gpus all --volumes-from luna_socia-ui-1:ro \
  -v "$repo/ui/app:/workspace/ui/app:ro" -v "$repo/model_evaluations:/workspace/model_evaluations:ro" \
  -v "$here:/workspace/notebooks" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e JUPYTER_TOKEN="$(cat "$here/.jupyter-token")" \
  -p 127.0.0.1:8888:8888 luna-socia-notebooks:local >/dev/null
echo "Jupyter is at http://127.0.0.1:8888/lab?token=$(cat "$here/.jupyter-token")"
