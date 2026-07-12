#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" == "0" ]]; then
  groupmod --non-unique --gid "${PGID:-1000}" appuser
  usermod --non-unique --uid "${PUID:-1000}" --gid "${PGID:-1000}" appuser
  mkdir -p /data/huggingface /data/jobs /data/checkpoints /outputs
  chown -R appuser:appuser /data /outputs
  exec gosu appuser "$@"
fi

exec "$@"
