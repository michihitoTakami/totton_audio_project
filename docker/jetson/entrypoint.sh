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
CONFIG_DIR="${MAGICBOX_CONFIG_DIR:-/opt/magicbox/config}"
CONFIG_FILE="${CONFIG_DIR}/config.json"
CONFIG_SYMLINK="${MAGICBOX_CONFIG_SYMLINK:-/opt/magicbox/config.json}"
DEFAULT_CONFIG="${MAGICBOX_DEFAULT_CONFIG:-/opt/magicbox/config-default/config.json}"
RESET_CONFIG="${MAGICBOX_RESET_CONFIG:-false}"
: "${MAGICBOX_ENABLE_RTP:=false}"
: "${MAGICBOX_RTP_AUTOSTART:=false}"

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

prepare_config() {
    mkdir -p "$CONFIG_DIR"

    if [[ ! -f "$DEFAULT_CONFIG" ]]; then
        log_error "Default config not found: $DEFAULT_CONFIG"
        exit 1
    fi

    local backup_path="${CONFIG_FILE}.bak"
    local reset_flag
    reset_flag="$(echo "$RESET_CONFIG" | tr '[:upper:]' '[:lower:]')"

    # Helper: validate JSON file, returns 0 if valid JSON
    validate_json() {
        local target="$1"
        if [[ ! -s "$target" ]]; then
            return 1
        fi
        jq empty "$target" >/dev/null 2>&1
    }

    # Helper: merge missing keys from DEFAULT_CONFIG into CONFIG_FILE (existing values win)
    merge_defaults() {
        local merged_path="${CONFIG_FILE}.merged"
        if jq -s '.[0] * .[1]' "$DEFAULT_CONFIG" "$CONFIG_FILE" > "$merged_path" 2>/dev/null; then
            mv -f "$merged_path" "$CONFIG_FILE"
        else
            rm -f "$merged_path" 2>/dev/null || true
            log_warn "Failed to merge defaults into config (keeping current config as-is)"
        fi
    }

    # Helper: Jetson migration - prefer I2S when both are enabled
    normalize_inputs() {
        if jq -e '.i2s.enabled == true and .loopback.enabled == true' "$CONFIG_FILE" >/dev/null 2>&1; then
            log_warn "Both i2s.enabled and loopback.enabled are true; disabling loopback (Jetson default)"
            local migrated_path="${CONFIG_FILE}.migrated"
            if jq '(.loopback.enabled = false)' "$CONFIG_FILE" > "$migrated_path" 2>/dev/null; then
                mv -f "$migrated_path" "$CONFIG_FILE"
            else
                rm -f "$migrated_path" 2>/dev/null || true
                log_warn "Failed to normalize input settings (daemon may refuse to start)"
            fi
        fi
    }

    # Optional reset via env
    if [[ "$reset_flag" == "true" || "$reset_flag" == "1" ]]; then
        log_warn "Reset requested via MAGICBOX_RESET_CONFIG, restoring default config"
        cp -f "$DEFAULT_CONFIG" "$CONFIG_FILE"
    else
        # Seed if missing/empty
        if [[ ! -s "$CONFIG_FILE" ]]; then
            log_info "Config not found, seeding default config to $CONFIG_FILE"
            cp -f "$DEFAULT_CONFIG" "$CONFIG_FILE"
        else
            # Validate existing JSON; if invalid, back up and restore defaults
            if validate_json "$CONFIG_FILE"; then
                log_info "Using existing config at $CONFIG_FILE"
            else
                log_warn "Config is invalid JSON, backing up to $backup_path and restoring default"
                cp -f "$CONFIG_FILE" "$backup_path" || true
                cp -f "$DEFAULT_CONFIG" "$CONFIG_FILE"
            fi
        fi
    fi

    # Always merge newly introduced default keys (e.g., i2s) into existing config safely.
    if validate_json "$CONFIG_FILE"; then
        merge_defaults
        normalize_inputs
    fi

    ln -sf "$CONFIG_FILE" "$CONFIG_SYMLINK"
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
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    exec "$DAEMON"
}

# Start both services (production mode)
start_all() {
    log_info "Starting Magic Box in production mode..."
    log_info "RTP enabled: ${MAGICBOX_ENABLE_RTP}"
    log_info "RTP autostart flag: ${MAGICBOX_RTP_AUTOSTART}"

    # Start daemon in background
    log_info "Starting Audio Daemon in background..."
    "$DAEMON" &
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
    config-init)
        prepare_config
        ;;
    web)
        # Web UI doesn't require GPU, skip nvidia check
        prepare_config
        start_web
        ;;
    daemon)
        check_nvidia
        check_audio
        prepare_config
        start_daemon
        ;;
    all)
        check_nvidia
        check_audio
        prepare_config
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
