#!/usr/bin/env bash
set -euo pipefail

# Allow full override
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

MODE=${PCM_BRIDGE_MODE:-rtp}
if [[ "$MODE" != "rtp" ]]; then
  echo "[entrypoint] MODE=$MODE はサポート外です。rtp を使用してください。" >&2
fi

args=(
  python3
  -m
  raspberry_pi.rtp_sender
  --device "${RTP_SENDER_DEVICE:-hw:0,0}"
  --host "${RTP_SENDER_HOST:-jetson}"
  --rtp-port "${RTP_SENDER_RTP_PORT:-46000}"
  --rtcp-port "${RTP_SENDER_RTCP_PORT:-46001}"
  --rtcp-listen-port "${RTP_SENDER_RTCP_LISTEN_PORT:-46002}"
  --payload-type "${RTP_SENDER_PAYLOAD_TYPE:-96}"
  --sample-rate "${RTP_SENDER_SAMPLE_RATE:-44100}"
  --channels "${RTP_SENDER_CHANNELS:-2}"
  --format "${RTP_SENDER_FORMAT:-S24_3BE}"
  --latency-ms "${RTP_SENDER_LATENCY_MS:-100}"
)

if [[ "${RTP_SENDER_DRY_RUN:-}" =~ ^(1|true|TRUE)$ ]]; then
  args+=("--dry-run")
fi

if [[ -n "${RTP_SENDER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${RTP_SENDER_EXTRA_ARGS})
  args+=("${extra_args[@]}")
fi

exec "${args[@]}"
