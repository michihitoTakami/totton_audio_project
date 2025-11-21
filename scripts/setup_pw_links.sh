#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_pw_links.sh [app_name]
# Connect an application's audio output to gpu_upsampler_sink.
# Default app_name=spotify.

APP_NAME="${1:-spotify}"

echo "Connecting $APP_NAME to GPU Upsampler..."

# Check if gpu_upsampler_sink exists
if ! pactl list short sinks 2>/dev/null | grep -q "gpu_upsampler_sink"; then
  echo "ERROR: gpu_upsampler_sink not found. Run restart_alsa_daemon.sh first." >&2
  exit 1
fi

# Check if GPU Upsampler Input exists (daemon running)
if ! pw-link -i 2>/dev/null | grep -q "GPU Upsampler Input:input_FL"; then
  echo "ERROR: GPU Upsampler Input not found. Is gpu_upsampler_alsa running?" >&2
  exit 1
fi

# Ensure sink monitor → GPU Upsampler links exist
pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL" 2>/dev/null || true
pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR" 2>/dev/null || true

# Try different port naming conventions for the app
connect_app() {
  local app="$1"
  local connected=0

  # Try common port patterns
  for suffix in "output_FL:output_FR" "output_0:output_1" "playback_FL:playback_FR"; do
    left="${suffix%%:*}"
    right="${suffix##*:}"

    if pw-link -o 2>/dev/null | grep -q "${app}:${left}"; then
      pw-link "${app}:${left}" gpu_upsampler_sink:playback_FL 2>/dev/null && connected=1
      pw-link "${app}:${right}" gpu_upsampler_sink:playback_FR 2>/dev/null || true
      break
    fi
  done

  return $((1 - connected))
}

if connect_app "$APP_NAME"; then
  echo "Connected: $APP_NAME → gpu_upsampler_sink → GPU Upsampler"
else
  echo "WARNING: Could not find ports for $APP_NAME"
  echo "Available output nodes:"
  pw-link -o | grep -E "^[^:]+:output" | cut -d: -f1 | sort -u | head -10
  echo ""
  echo "Manual connection: pw-link <app>:output_FL gpu_upsampler_sink:playback_FL"
fi
