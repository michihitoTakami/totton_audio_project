#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="/tmp/gpu_upsampler_alsa.log"
BINARY="$ROOT_DIR/build/gpu_upsampler_alsa"

# Default ALSA device (SMSL DAC)
DEFAULT_ALSA_DEVICE="hw:AUDIO,0"

# Kill existing processes (use -9 to ensure termination)
pkill -9 -f gpu_upsampler_alsa 2>/dev/null || true
sleep 0.5

# Resolve binary path
if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

# Determine ALSA device
DEVICE_ARG="${1:-}"
if [[ -n "$DEVICE_ARG" ]]; then
  export ALSA_DEVICE="$DEVICE_ARG"
else
  # Auto-detect SMSL DAC card number
  SMSL_CARD=$(aplay -l 2>/dev/null | grep -i "SMSL" | head -1 | sed -n 's/^カード \([0-9]*\):.*/\1/p')
  if [[ -z "$SMSL_CARD" ]]; then
    SMSL_CARD=$(aplay -l 2>/dev/null | grep -i "SMSL" | head -1 | sed -n 's/^card \([0-9]*\):.*/\1/p')
  fi
  if [[ -n "$SMSL_CARD" ]]; then
    export ALSA_DEVICE="hw:${SMSL_CARD},0"
  else
    export ALSA_DEVICE="$DEFAULT_ALSA_DEVICE"
  fi
fi

echo "ALSA device: $ALSA_DEVICE"

# Release SMSL DAC from PipeWire by setting card profile to 'off'
SMSL_CARD_NAME=$(pactl list short cards 2>/dev/null | grep -i "SMSL" | awk '{print $2}')
if [[ -n "$SMSL_CARD_NAME" ]]; then
  echo "Releasing $SMSL_CARD_NAME from PipeWire..."
  pactl set-card-profile "$SMSL_CARD_NAME" off 2>/dev/null || true
  sleep 0.3
fi

# Create gpu_upsampler_sink if not present (using pactl for reliability)
if ! pactl list short sinks 2>/dev/null | grep -q "gpu_upsampler_sink"; then
  echo "Creating gpu_upsampler_sink..."
  pactl load-module module-null-sink sink_name=gpu_upsampler_sink sink_properties=device.description="GPU_Upsampler_Sink" >/dev/null
  sleep 0.3
fi

# Start daemon
echo "Starting gpu_upsampler_alsa..."
nohup "$BINARY" > "$LOG_FILE" 2>&1 &
PID=$!
sleep 1.5

# Check if process is still running
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: Daemon failed to start. Check log:" >&2
  tail -n 20 "$LOG_FILE"
  exit 1
fi

echo "Started PID=$PID. Log: $LOG_FILE"
tail -n 20 "$LOG_FILE" || true

# Setup PipeWire links (sink monitor → GPU Upsampler Input)
echo ""
echo "Setting up PipeWire links..."
sleep 0.5

# Wait for GPU Upsampler Input ports to appear
for i in {1..10}; do
  if pw-link -i 2>/dev/null | grep -q "GPU Upsampler Input:input_FL"; then
    break
  fi
  sleep 0.3
done

# Create links
pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL" 2>/dev/null || true
pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR" 2>/dev/null || true

echo "Links configured. Ready to use."
echo ""
echo "To route audio: Set 'GPU Upsampler Sink' as output in sound settings,"
echo "or run: pw-link <app>:output_FL gpu_upsampler_sink:playback_FL"
