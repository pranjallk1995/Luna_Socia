#!/usr/bin/env bash
# Gets the notebooks' container ready beside the running Luna Socia. VS Code runs this itself before
# it opens the repo in the container (.devcontainer/devcontainer.json).
#   ./notebooks/start.sh prepare   Socia's settings and the image (what VS Code runs)
#   ./notebooks/start.sh run       the container without VS Code, for `docker exec` (model evaluations)
#   ./notebooks/start.sh stop      stop it
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
name=luna_socia-notebooks
case "${1:-prepare}" in
  stop) docker rm -f "$name" >/dev/null && echo "Stopped."; exit 0 ;;
  prepare|run) ;;
  *) echo "Usage: $0 [prepare|run|stop]"; exit 2 ;;
esac
docker inspect luna_socia-ui-1 >/dev/null 2>&1 || { echo "Start Luna Socia first."; exit 1; }
umask 077
# Socia's settings as the running app has them (with secrets: kept out of git, readable by you only).
docker inspect luna_socia-ui-1 --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -vE '^(HOME|PATH|PYTHONPATH)=' > "$here/.socia.env"
umask 022
# A container left by `run` would take the name VS Code's container needs.
if [ "${1:-prepare}" = prepare ] && [ -z "$(docker inspect "$name" --format '{{index .Config.Labels "devcontainer.local_folder"}}' 2>/dev/null)" ]; then
  docker rm -f "$name" >/dev/null 2>&1 || true
fi
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t luna-socia-notebooks:local "$here" >/dev/null
[ "${1:-prepare}" = run ] || exit 0
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --network luna_sentia_network --env-file "$here/.socia.env" \
  --gpus all --volumes-from luna_socia-ui-1:ro -v "$repo:/workspace" luna-socia-notebooks:local >/dev/null
echo "Running: docker exec -it $name python ..."
