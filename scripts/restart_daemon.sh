#!/bin/bash
# GPU Upsampler Daemon restart script
# Usage: ./scripts/restart_daemon.sh [config_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DAEMON_NAME="gpu_upsampler_alsa"
DAEMON_PATH="$PROJECT_ROOT/build/$DAEMON_NAME"
CONFIG_PATH="${1:-$PROJECT_ROOT/config.json}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if daemon binary exists
if [ ! -f "$DAEMON_PATH" ]; then
    log_error "Daemon binary not found: $DAEMON_PATH"
    log_info "Run 'cmake --build build' first"
    exit 1
fi

# Find and kill existing daemon process
kill_existing() {
    local pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "Stopping existing daemon (PID: $pids)..."
        kill $pids 2>/dev/null || true
        sleep 1
        # Force kill if still running
        pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            log_warn "Force killing daemon..."
            kill -9 $pids 2>/dev/null || true
            sleep 0.5
        fi
    fi
}

# Start daemon
start_daemon() {
    log_info "Starting daemon..."
    cd "$PROJECT_ROOT"
    nohup "$DAEMON_PATH" > /tmp/gpu_upsampler.log 2>&1 &
    local pid=$!
    sleep 1

    if kill -0 $pid 2>/dev/null; then
        log_info "Daemon started (PID: $pid)"
        log_info "Log: /tmp/gpu_upsampler.log"
    else
        log_error "Daemon failed to start. Check log:"
        tail -20 /tmp/gpu_upsampler.log
        exit 1
    fi
}

# Send SIGHUP for config reload (if daemon supports it)
reload_config() {
    local pids=$(pgrep -f "$DAEMON_NAME" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "Sending SIGHUP to daemon (PID: $pids) for config reload..."
        kill -HUP $pids
        log_info "Config reload signal sent"
    else
        log_warn "Daemon not running"
        return 1
    fi
}

# Main
case "${2:-restart}" in
    start)
        kill_existing
        start_daemon
        ;;
    stop)
        kill_existing
        log_info "Daemon stopped"
        ;;
    reload)
        reload_config
        ;;
    restart|*)
        kill_existing
        start_daemon
        ;;
esac
