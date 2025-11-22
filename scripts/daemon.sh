#!/usr/bin/env bash
set -euo pipefail

# GPU Upsampler Daemon Control Script
# Usage: ./scripts/daemon.sh [start|stop|restart|status] [eq_profile|off]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="$ROOT_DIR/build/gpu_upsampler_alsa"
CONFIG_FILE="$ROOT_DIR/config.json"
LOG_FILE="/tmp/gpu_upsampler_alsa.log"
PID_FILE="/tmp/gpu_upsampler_alsa.pid"
DAEMON_NAME="gpu_upsampler_alsa"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Verify PID belongs to our daemon (guard against PID reuse)
# Returns 0 if valid, 1 if not our process
verify_daemon_pid() {
    local pid="$1"
    [[ -z "$pid" ]] && return 1

    # Check process exists
    kill -0 "$pid" 2>/dev/null || return 1

    # Verify process name via /proc/<pid>/comm
    # Note: comm is truncated to 15 chars, so "gpu_upsampler_alsa" -> "gpu_upsampler_a"
    local comm_file="/proc/$pid/comm"
    if [[ -f "$comm_file" ]]; then
        local comm
        comm=$(cat "$comm_file" 2>/dev/null || true)
        # Check if comm starts with "gpu_upsampler" (handles truncation)
        if [[ "$comm" != gpu_upsampler* ]]; then
            return 1  # PID was reused by another process
        fi
    fi

    return 0
}

# Kill existing daemon using PID file
kill_daemon() {
    local pid=""

    # First, try PID file
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if verify_daemon_pid "$pid"; then
            log_info "Stopping daemon (PID: $pid from PID file)..."
            kill "$pid" 2>/dev/null || true
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 0.3
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "Force killing PID $pid..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 0.5
            fi
        elif [[ -n "$pid" ]]; then
            log_warn "PID file contains stale PID $pid (not our daemon)"
        fi
        rm -f "$PID_FILE"
    fi

    # Fallback: kill any remaining processes by name (safety net)
    local pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log_warn "Found orphan processes: $pids (killing...)"
        kill $pids 2>/dev/null || true
        sleep 0.5
        pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill -9 $pids 2>/dev/null || true
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
    rm -f "$LOG_FILE"  # Clear old log
    nohup "$BINARY" > "$LOG_FILE" 2>&1 &

    # Wait for daemon to create PID file and initialize
    local pid=""
    for i in {1..30}; do
        sleep 0.5
        # Check if PID file was created by daemon
        if [[ -f "$PID_FILE" ]]; then
            pid=$(cat "$PID_FILE" 2>/dev/null || true)
            if verify_daemon_pid "$pid"; then
                # Check if ALSA is configured
                if grep -q "ALSA: Output device configured" "$LOG_FILE" 2>/dev/null; then
                    break
                fi
            fi
        fi
        # Check for early crash (e.g., another instance running)
        if grep -q "Error: Another instance is already running" "$LOG_FILE" 2>/dev/null; then
            log_error "Another daemon instance is already running. Log:"
            tail -10 "$LOG_FILE"
            exit 1
        fi
    done

    # Verify daemon is running
    if ! verify_daemon_pid "$pid"; then
        log_error "Daemon failed to start. Log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi

    log_info "Daemon started (PID: $pid)"
    setup_links
    log_info "Ready. Log: $LOG_FILE"
}

# Show status using PID file
show_status() {
    local pid=""
    local running=false

    # Check PID file
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if verify_daemon_pid "$pid"; then
            running=true
            log_info "Daemon running (PID: $pid)"
        elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log_warn "PID file contains PID $pid but it's not our daemon (PID reuse)"
        else
            log_warn "Stale PID file found (PID: $pid not running)"
        fi
    fi

    # Check for orphan processes
    local pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        if [[ "$running" == "false" ]]; then
            log_warn "Orphan daemon processes found: $pids (no PID file)"
        fi
    elif [[ "$running" == "false" ]]; then
        log_warn "Daemon not running"
    fi

    # Show audio sinks
    if [[ "$running" == "true" ]] || [[ -n "$pids" ]]; then
        echo "Audio sinks:"
        pactl list short sinks 2>/dev/null | grep -E "gpu_upsampler|easyeffects" || true
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
