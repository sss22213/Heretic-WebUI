#!/usr/bin/env bash
# RunPod entrypoint: run Ollama and the WebUI in one container, both writing
# to the persistent /workspace volume. Runs as root (RunPod convention); the
# gosu/PUID user drop from docker/entrypoint.sh is intentionally not used.
set -euo pipefail

mkdir -p "${APP_DATA_DIR}/jobs" "${APP_DATA_DIR}/checkpoints" \
    "${APP_OUTPUT_DIR}" "${APP_MODELS_DIR}" "${OLLAMA_MODELS}" "${HF_HOME}"

echo "[runpod] starting ollama serve (models: ${OLLAMA_MODELS})"
ollama serve &
OLLAMA_PID=$!

for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null; then
    echo "[runpod] ollama is up: $(curl -sf http://127.0.0.1:11434/api/version)"
    break
  fi
  sleep 1
done
if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null; then
  echo "[runpod] WARNING: ollama did not become ready in 60s; WebUI starts anyway" >&2
fi

echo "[runpod] starting Heretic WebUI on :8000"
cd /app
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
UVICORN_PID=$!

term() { kill -TERM "$OLLAMA_PID" "$UVICORN_PID" 2>/dev/null || true; }
trap term TERM INT

# Exit when either process dies so RunPod restarts the whole pod consistently.
set +e
wait -n "$OLLAMA_PID" "$UVICORN_PID"
STATUS=$?
term
wait "$OLLAMA_PID" "$UVICORN_PID" 2>/dev/null || true
exit "$STATUS"
