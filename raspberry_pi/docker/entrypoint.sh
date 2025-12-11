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

CHRT_PRIO=${RTP_SENDER_CHRT_PRIO:-20}
NICE_LEVEL=${RTP_SENDER_NICE_LEVEL:--11}
launcher=()

# RT優先度が使えそうならchrtを優先し、だめならniceにフォールバック
if command -v chrt >/dev/null 2>&1 && [[ "${CHRT_PRIO}" =~ ^[0-9]+$ ]] && (( CHRT_PRIO > 0 )); then
  if chrt -f "${CHRT_PRIO}" true 2>/dev/null; then
    launcher=(chrt -f "${CHRT_PRIO}")
  fi
fi

if [[ ${#launcher[@]} -eq 0 ]] && command -v nice >/dev/null 2>&1 && [[ "${NICE_LEVEL}" =~ ^-?[0-9]+$ ]]; then
  if nice -n "${NICE_LEVEL}" true 2>/dev/null; then
    launcher=(nice -n "${NICE_LEVEL}")
  fi
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

exec "${launcher[@]}" "${args[@]}"
