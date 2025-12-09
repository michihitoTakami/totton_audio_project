#!/usr/bin/env bash
set -euo pipefail

# Allow full override
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

MODE=${PCM_BRIDGE_MODE:-run}

if [[ "$MODE" == "help" ]]; then
  exec /usr/local/bin/rpi_pcm_bridge --help
elif [[ "$MODE" == "version" ]]; then
  exec /usr/local/bin/rpi_pcm_bridge --version
fi

args=(
  --device "${PCM_BRIDGE_DEVICE:-hw:0,0}"
  --host "${PCM_BRIDGE_HOST:-127.0.0.1}"
  --port "${PCM_BRIDGE_PORT:-46001}"
  --rate "${PCM_BRIDGE_RATE:-48000}"
  --format "${PCM_BRIDGE_FORMAT:-S16_LE}"
  --frames "${PCM_BRIDGE_FRAMES:-4096}"
  --log-level "${PCM_BRIDGE_LOG_LEVEL:-warn}"
)

if [[ -n "${PCM_BRIDGE_ITERATIONS:-}" ]]; then
  args+=(--iterations "${PCM_BRIDGE_ITERATIONS}")
fi

if [[ -n "${PCM_BRIDGE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${PCM_BRIDGE_EXTRA_ARGS})
  args+=("${extra_args[@]}")
fi

exec /usr/local/bin/rpi_pcm_bridge "${args[@]}"
