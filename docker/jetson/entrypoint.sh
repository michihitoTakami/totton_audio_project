#!/bin/bash
#
# Magic Box Container Entrypoint
#
# Usage:
#   docker run magicbox web       # Start Web UI only
#   docker run magicbox daemon    # Start Audio Daemon only
#   docker run magicbox all       # Start both (production)
#   docker run magicbox bash      # Interactive shell
#

set -e

# Paths
UVICORN="/opt/magicbox/venv/bin/uvicorn"
DAEMON="/opt/magicbox/bin/gpu_upsampler_alsa"
CONFIG="/opt/magicbox/config.json"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check NVIDIA runtime
check_nvidia() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        log_info "NVIDIA GPU detected via nvidia-smi: ${gpu_name:-unknown}"
        return
    fi

    if command -v nvidia-container-cli >/dev/null 2>&1; then
        local runtime_ver
        runtime_ver=$(nvidia-container-cli --version 2>/dev/null | head -1)
        log_info "NVIDIA container runtime detected: ${runtime_ver:-unknown}"
        return
    fi

    if [ -f /proc/device-tree/gpu/compatible ]; then
        local jetson_gpu
        jetson_gpu=$(tr -d '\0' < /proc/device-tree/gpu/compatible 2>/dev/null)
        log_info "Jetson GPU detected: ${jetson_gpu:-unknown}"
        return
    fi

    if command -v tegrastats >/dev/null 2>&1; then
        log_info "Jetson tegrastats available - assuming GPU runtime is ready"
        return
    fi

    log_error "NVIDIA runtime not available!"
    log_error "Ensure --runtime=nvidia (or Jetson default runtime) is configured."
    exit 1
}

# Check audio devices
check_audio() {
    if [ ! -d "/dev/snd" ]; then
        log_warn "Audio devices not mounted. Run with: --device /dev/snd"
    else
        log_info "Audio devices available"
    fi
}

# Start Web UI (FastAPI)
start_web() {
    log_info "Starting Web UI on port 80..."
    exec "$UVICORN" web.main:app --host 0.0.0.0 --port 80
}

# Start Audio Daemon
start_daemon() {
    log_info "Starting Audio Daemon..."
    if [ ! -f "$CONFIG" ]; then
        log_error "Config file not found: $CONFIG"
        exit 1
    fi
    exec "$DAEMON" --config "$CONFIG"
}

# Start both services (production mode)
start_all() {
    log_info "Starting Magic Box in production mode..."

    # Start daemon in background
    log_info "Starting Audio Daemon in background..."
    "$DAEMON" --config "$CONFIG" &
    DAEMON_PID=$!

    # Wait a bit for daemon to initialize
    sleep 2

    # Check if daemon is running
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        log_error "Audio Daemon failed to start!"
        exit 1
    fi

    log_info "Audio Daemon started (PID: $DAEMON_PID)"

    # Start Web UI in foreground
    log_info "Starting Web UI..."
    "$UVICORN" web.main:app --host 0.0.0.0 --port 80 &
    WEB_PID=$!

    # Trap signals for graceful shutdown
    trap 'log_info "Shutting down..."; kill "$DAEMON_PID" "$WEB_PID" 2>/dev/null; exit 0' SIGTERM SIGINT

    # Wait for either process to exit
    wait -n "$DAEMON_PID" "$WEB_PID"
    EXIT_CODE=$?

    log_warn "A process exited with code $EXIT_CODE"
    kill "$DAEMON_PID" "$WEB_PID" 2>/dev/null || true
    exit "$EXIT_CODE"
}

# Main
case "${1:-web}" in
    web)
        # Web UI doesn't require GPU, skip nvidia check
        start_web
        ;;
    daemon)
        check_nvidia
        check_audio
        start_daemon
        ;;
    all)
        check_nvidia
        check_audio
        start_all
        ;;
    bash|sh)
        exec /bin/bash
        ;;
    *)
        # Pass through to exec
        exec "$@"
        ;;
esac
