#!/usr/bin/env bash
set -euo pipefail

# GPU Upsampler Daemon Control Script
# Usage: ./scripts/daemon.sh [start|stop|restart|status] [eq_profile|off]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="$ROOT_DIR/build/gpu_upsampler_alsa"
CONFIG_FILE="$ROOT_DIR/config.json"
LOG_FILE="/tmp/gpu_upsampler_alsa.log"
DAEMON_NAME="gpu_upsampler_alsa"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Kill existing daemon
kill_daemon() {
    local pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log_info "Stopping daemon (PID: $pids)..."
        kill $pids 2>/dev/null || true
        sleep 1
        pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            log_warn "Force killing..."
            kill -9 $pids 2>/dev/null || true
            sleep 0.5
        fi
    fi
}

# Setup PipeWire links
setup_links() {
    log_info "Setting up PipeWire links..."
    for i in {1..10}; do
        if pw-link -i 2>/dev/null | grep -q "GPU Upsampler Input:input_FL"; then
            break
        fi
        sleep 0.3
    done
    pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL" 2>/dev/null || true
    pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR" 2>/dev/null || true
    log_info "Links configured"
}

# Start daemon
start_daemon() {
    if [[ ! -x "$BINARY" ]]; then
        log_error "Binary not found: $BINARY"
        log_info "Run 'cmake --build build' first"
        exit 1
    fi

    # Handle EQ argument
    local eq_arg="${1:-}"
    if [[ -n "$eq_arg" ]]; then
        if [[ "$eq_arg" == "off" ]]; then
            log_info "EQ: disabled"
            jq '.eqEnabled=false' "$CONFIG_FILE" > /tmp/cfg_tmp && mv /tmp/cfg_tmp "$CONFIG_FILE"
        elif [[ -f "$eq_arg" ]] || [[ -f "$ROOT_DIR/$eq_arg" ]]; then
            local eq_path="$eq_arg"
            [[ -f "$ROOT_DIR/$eq_arg" ]] && eq_path="$ROOT_DIR/$eq_arg"
            eq_path="$(realpath "$eq_path")"
            log_info "EQ: $eq_path"
            jq --arg p "$eq_path" '.eqEnabled=true | .eqProfilePath=$p' "$CONFIG_FILE" > /tmp/cfg_tmp && mv /tmp/cfg_tmp "$CONFIG_FILE"
        fi
    fi

    # Create sink if needed
    if ! pactl list short sinks 2>/dev/null | grep -q "gpu_upsampler_sink"; then
        log_info "Creating gpu_upsampler_sink..."
        pactl load-module module-null-sink sink_name=gpu_upsampler_sink sink_properties=device.description="GPU_Upsampler_Sink" >/dev/null
        sleep 0.3
    fi

    # Start
    log_info "Starting daemon..."
    cd "$ROOT_DIR"
    nohup "$BINARY" > "$LOG_FILE" 2>&1 &
    local pid=$!

    # Wait for initialization
    for i in {1..15}; do
        sleep 0.5
        if ! kill -0 "$pid" 2>/dev/null; then
            log_error "Daemon crashed. Log:"
            tail -20 "$LOG_FILE"
            exit 1
        fi
        if grep -q "ALSA: Output device configured" "$LOG_FILE" 2>/dev/null; then
            break
        fi
    done

    if ! kill -0 "$pid" 2>/dev/null; then
        log_error "Daemon failed to start. Log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi

    log_info "Daemon started (PID: $pid)"
    setup_links
    log_info "Ready. Log: $LOG_FILE"
}

# Show status
show_status() {
    local pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log_info "Daemon running (PID: $pids)"
        pactl list short sinks 2>/dev/null | grep -E "gpu_upsampler|easyeffects" || true
    else
        log_warn "Daemon not running"
    fi
}

# Main
case "${1:-restart}" in
    start)
        kill_daemon
        start_daemon "${2:-}"
        ;;
    stop)
        kill_daemon
        log_info "Daemon stopped"
        ;;
    restart)
        kill_daemon
        start_daemon "${2:-}"
        ;;
    status)
        show_status
        ;;
    links)
        setup_links
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status|links] [eq_profile|off]"
        exit 1
        ;;
esac
