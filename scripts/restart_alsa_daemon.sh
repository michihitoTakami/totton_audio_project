#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="/tmp/gpu_upsampler_alsa.log"
BINARY="$ROOT_DIR/build/gpu_upsampler_alsa"

# Default ALSA device (SMSL DAC)
DEFAULT_ALSA_DEVICE="hw:AUDIO,0"

# Save current default sink and card profile for restoration on failure
# Skip virtual sinks (gpu_upsampler_sink, easyeffects_sink, etc.) and find real hardware sink
ORIGINAL_DEFAULT_SINK=$(pactl info 2>/dev/null | grep "デフォルトシンク:" | cut -d: -f2 | tr -d ' ')
if [[ -z "$ORIGINAL_DEFAULT_SINK" ]]; then
  ORIGINAL_DEFAULT_SINK=$(pactl info 2>/dev/null | grep "Default Sink:" | cut -d: -f2 | tr -d ' ')
fi

# If default sink is virtual, find a real ALSA output sink instead
if [[ "$ORIGINAL_DEFAULT_SINK" == "gpu_upsampler_sink" || "$ORIGINAL_DEFAULT_SINK" == "easyeffects_sink" || "$ORIGINAL_DEFAULT_SINK" != alsa_output* ]]; then
  ORIGINAL_DEFAULT_SINK=$(pactl list short sinks 2>/dev/null | grep "^[0-9]*	alsa_output" | grep -v "hdmi" | awk '{print $2}' | head -1)
  if [[ -z "$ORIGINAL_DEFAULT_SINK" ]]; then
    # Fallback: any alsa_output
    ORIGINAL_DEFAULT_SINK=$(pactl list short sinks 2>/dev/null | grep "^[0-9]*	alsa_output" | awk '{print $2}' | head -1)
  fi
fi
ORIGINAL_CARD_NAME=""
ORIGINAL_CARD_PROFILE=""

# Function to restore original audio output
restore_original_audio() {
  echo ""
  echo "Restoring original audio output..."
  if [[ -n "$ORIGINAL_CARD_NAME" && -n "$ORIGINAL_CARD_PROFILE" ]]; then
    pactl set-card-profile "$ORIGINAL_CARD_NAME" "$ORIGINAL_CARD_PROFILE" 2>/dev/null || true
    sleep 0.5
  fi
  if [[ -n "$ORIGINAL_DEFAULT_SINK" ]]; then
    # Wait for sink to reappear after profile restore
    for i in {1..10}; do
      if pactl list short sinks 2>/dev/null | grep -q "$ORIGINAL_DEFAULT_SINK"; then
        pactl set-default-sink "$ORIGINAL_DEFAULT_SINK" 2>/dev/null || true
        echo "Default sink restored to: $ORIGINAL_DEFAULT_SINK"
        return
      fi
      sleep 0.3
    done
    echo "Warning: Could not restore original sink $ORIGINAL_DEFAULT_SINK"
  fi
}

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

# Find the card associated with the original default sink and save its profile
if [[ -n "$ORIGINAL_DEFAULT_SINK" ]]; then
  # Extract card name from sink name (e.g., alsa_output.usb-XXX.analog-stereo -> alsa_card.usb-XXX)
  ORIGINAL_CARD_NAME=$(echo "$ORIGINAL_DEFAULT_SINK" | sed 's/alsa_output\./alsa_card./' | sed 's/\.analog-stereo$//' | sed 's/\.hdmi-stereo.*$//')
  if pactl list short cards 2>/dev/null | grep -q "$ORIGINAL_CARD_NAME"; then
    ORIGINAL_CARD_PROFILE=$(pactl list cards 2>/dev/null | grep -A 55 "$ORIGINAL_CARD_NAME" | grep "有効なプロフィール:" | head -1 | sed 's/.*有効なプロフィール: *//')
    if [[ -z "$ORIGINAL_CARD_PROFILE" ]]; then
      ORIGINAL_CARD_PROFILE=$(pactl list cards 2>/dev/null | grep -A 55 "$ORIGINAL_CARD_NAME" | grep "Active Profile:" | head -1 | sed 's/.*Active Profile: *//')
    fi
    echo "Saved original card: $ORIGINAL_CARD_NAME (profile: $ORIGINAL_CARD_PROFILE)"
    echo "Releasing $ORIGINAL_CARD_NAME from PipeWire..."
    pactl set-card-profile "$ORIGINAL_CARD_NAME" off 2>/dev/null || true
    sleep 0.3
  fi
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

# Wait for streaming to start and check for early crash
echo "Waiting for daemon to initialize..."
for i in {1..10}; do
  sleep 0.5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: Daemon crashed during startup. Check log:" >&2
    tail -n 30 "$LOG_FILE"
    restore_original_audio
    exit 1
  fi
  # Check if ALSA output is configured (initialization complete)
  if grep -q "ALSA: Output device configured" "$LOG_FILE" 2>/dev/null; then
    # Wait longer to catch streaming-phase crashes (CUDA errors happen when audio starts flowing)
    echo "ALSA configured. Testing stability..."
    for j in {1..8}; do
      sleep 0.5
      if ! kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Daemon crashed after ALSA setup. Check log:" >&2
        tail -n 30 "$LOG_FILE"
        restore_original_audio
        exit 1
      fi
      # Also check for CUDA errors in log
      if grep -q "CUDA Error" "$LOG_FILE" 2>/dev/null; then
        sleep 0.5  # Give process time to crash
        if ! kill -0 "$PID" 2>/dev/null; then
          echo "ERROR: CUDA error detected. Check log:" >&2
          tail -n 30 "$LOG_FILE"
          restore_original_audio
          exit 1
        fi
      fi
    done
    break
  fi
done

# Final check
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: Daemon is not running. Check log:" >&2
  tail -n 30 "$LOG_FILE"
  restore_original_audio
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

echo "Links configured."

# Test audio streaming to detect CUDA errors
echo "Testing audio streaming..."
timeout 1 pw-play --target=gpu_upsampler_sink /usr/share/sounds/freedesktop/stereo/message.oga 2>/dev/null &
sleep 2

# Check for crash after audio test
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: Daemon crashed during audio test. Check log:" >&2
  tail -n 30 "$LOG_FILE"
  restore_original_audio
  exit 1
fi

# Check for CUDA errors
if grep -q "CUDA Error" "$LOG_FILE" 2>/dev/null; then
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: CUDA error detected during audio test. Check log:" >&2
    tail -n 30 "$LOG_FILE"
    restore_original_audio
    exit 1
  fi
fi

echo "Ready to use."
echo ""
echo "To route audio: Set 'GPU Upsampler Sink' as output in sound settings,"
echo "or run: pw-link <app>:output_FL gpu_upsampler_sink:playback_FL"
