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
elif [[ "$MODE" == "rtp-help" ]]; then
  exec /usr/local/bin/rpi_rtp_sender --help
elif [[ "$MODE" == "rtp-version" ]]; then
  exec /usr/local/bin/rpi_rtp_sender --version
elif [[ "$MODE" == "rtp" ]]; then
  rtp_args=(
    --device "${RTP_SENDER_DEVICE:-hw:0,0}"
    --host "${RTP_SENDER_HOST:-127.0.0.1}"
    --rtp-port "${RTP_SENDER_RTP_PORT:-46000}"
    --rtcp-port "${RTP_SENDER_RTCP_PORT:-46001}"
    --rtcp-listen-port "${RTP_SENDER_RTCP_LISTEN_PORT:-46002}"
    --payload-type "${RTP_SENDER_PAYLOAD_TYPE:-96}"
    --poll-ms "${RTP_SENDER_POLL_MS:-250}"
    --log-level "${RTP_SENDER_LOG_LEVEL:-warn}"
  )

  if [[ -n "${RTP_SENDER_FORMAT:-}" ]]; then
    rtp_args+=(--format "${RTP_SENDER_FORMAT}")
  fi
  if [[ -n "${RTP_SENDER_NOTIFY_URL:-}" ]]; then
    rtp_args+=(--rate-notify-url "${RTP_SENDER_NOTIFY_URL}")
  fi
  if [[ "${RTP_SENDER_DRY_RUN:-}" == "1" || "${RTP_SENDER_DRY_RUN:-}" == "true" ]]; then
    rtp_args+=(--dry-run)
  fi
  if [[ -n "${RTP_SENDER_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${RTP_SENDER_EXTRA_ARGS})
    rtp_args+=("${extra_args[@]}")
  fi

  exec /usr/local/bin/rpi_rtp_sender "${rtp_args[@]}"
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
