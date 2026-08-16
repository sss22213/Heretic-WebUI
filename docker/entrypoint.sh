#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" == "0" ]]; then
  groupmod --non-unique --gid "${PGID:-1000}" appuser
  usermod --non-unique --uid "${PUID:-1000}" --gid "${PGID:-1000}" appuser
  mkdir -p /data/huggingface /data/jobs /data/checkpoints /outputs
  chown -R appuser:appuser /data /outputs

  # Cloud hosts (e.g. RunPod) inject the account SSH key as PUBLIC_KEY; start
  # sshd only then, so SSH tunnels and scp work without exposing the web port.
  if [[ -n "${PUBLIC_KEY:-}" ]] && command -v /usr/sbin/sshd >/dev/null; then
    mkdir -p /root/.ssh /run/sshd
    printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd
  fi

  exec gosu appuser "$@"
fi

exec "$@"
