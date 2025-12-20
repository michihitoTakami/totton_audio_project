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
PROFILE="${MAGICBOX_PROFILE:-base}"
DATA_ROOT="${GPU_OS_DATA_DIR:-${DATA_DIR:-/var/lib/gpu_upsampler}}"
OPRA_CACHE_DIR="${DATA_ROOT}/opra"
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

prepare_opra_cache_dir() {
    # Ensure OPRA cache layout exists on the mounted volume with writable perms
    mkdir -p "${OPRA_CACHE_DIR}/versions" "${OPRA_CACHE_DIR}/lock"
    chmod 775 "${OPRA_CACHE_DIR}" || true
    chown -R magicbox:magicbox "${OPRA_CACHE_DIR}" 2>/dev/null || true
}

wait_for_alsa() {
    local timeout="${MAGICBOX_WAIT_AUDIO_SECS:-0}"
    if [[ -z "${timeout}" ]]; then
        timeout=0
    fi
    if [[ "${timeout}" == "0" ]]; then
        return 0
    fi
    log_info "Waiting for ALSA devices (timeout=${timeout}s)..."
    local start
    start="$(date +%s)"
    while true; do
        # /proc/asound はコンテナ環境によっては見えない/空になることがあるため、
        # /dev/snd のデバイスノードを主判定にする。
        if [[ -d "/dev/snd" ]]; then
            if ls /dev/snd/controlC* >/dev/null 2>&1; then
                log_info "ALSA devices detected (/dev/snd/controlC*)"
                return 0
            fi
        fi
        local now
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            log_warn "Timed out waiting for ALSA devices"
            return 0
        fi
        sleep 1
    done
}

_find_card_index_by_id() {
    local card_id="$1"
    if [[ -z "${card_id}" ]]; then
        return 1
    fi
    if [[ ! -r "/proc/asound/cards" ]]; then
        return 1
    fi
    awk -v id="${card_id}" '
        $0 ~ "\\[" id "\\]" {print $1; exit 0}
        END {exit 1}
    ' /proc/asound/cards 2>/dev/null
}

_amixer_try_card() {
    local card="$1"
    if [[ -z "${card}" ]]; then
        return 1
    fi
    # Avoid hangs (rt driver / kernel quirks) - best effort.
    timeout 2 amixer -c "${card}" controls >/dev/null 2>&1
}

configure_jetson_ape_i2s() {
    # Best-effort only.
    # First priority: do the simple thing (explicit APE card) and just log failures.
    if ! command -v amixer >/dev/null 2>&1; then
        log_warn "amixer not available; skipping APE/I2S routing"
        return 0
    fi
    local ape="${MAGICBOX_APE_CARD:-APE}"

    log_info "Applying Jetson APE/I2S routing (card=${ape})..."

    # NOTE: First, apply routing.
    if timeout 2 amixer -c "${ape}" cset name="ADMAIF1 Mux" "I2S2" >/dev/null 2>&1; then
        log_info "amixer: ADMAIF1 Mux = I2S2"
    else
        log_warn "amixer failed: ADMAIF1 Mux = I2S2 (skipped)"
        timeout 2 amixer -c "${ape}" cset name="ADMAIF1 Mux" "I2S2" || true
    fi

    # Then, just report current modes (debug visibility).
    timeout 2 amixer -c "${ape}" cget name="I2S2 codec master mode" || true
    timeout 2 amixer -c "${ape}" cget name="I2S2 codec frame mode" || true
}

prepare_config() {
    prepare_opra_cache_dir
    mkdir -p "$CONFIG_DIR"

    if [[ ! -f "$DEFAULT_CONFIG" ]]; then
        log_error "Default config not found: $DEFAULT_CONFIG"
        exit 1
    fi

    local backup_path="${CONFIG_FILE}.bak"
    local reset_flag
    reset_flag="$(echo "$RESET_CONFIG" | tr '[:upper:]' '[:lower:]')"
    local apply_profile_on_seed=false

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

    apply_profile() {
        if [[ "${PROFILE}" != "jetson" ]]; then
            return 0
        fi
        log_info "Applying config profile: jetson"
        local profiled_path="${CONFIG_FILE}.profile"
        if jq '
            .gain = 1.0
            | .eqEnabled = false
            | .eqProfilePath = ""
            | .phaseType = "minimum"
            | .filterPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin"
            | .filterPath44kMin = "data/coefficients/filter_44k_16x_2m_min_phase.bin"
            | .filterPath48kMin = "data/coefficients/filter_48k_16x_2m_min_phase.bin"
            | .i2s.enabled = true
            | .i2s.device = "hw:APE,0"
            | .i2s.sampleRate = 0
            | .i2s.channels = 2
            | .i2s.format = "S32_LE"
            | .i2s.periodFrames = 1024
            | .loopback.enabled = false
        ' "$CONFIG_FILE" > "$profiled_path" 2>/dev/null; then
            mv -f "$profiled_path" "$CONFIG_FILE"
        else
            rm -f "$profiled_path" 2>/dev/null || true
            log_warn "Failed to apply jetson profile (keeping current config as-is)"
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

        # Jetson migration: old default used hw:APE,0,0 but field reports show hw:APE,0 is more stable.
        # Only rewrite when the value is exactly the old default (user overrides are kept).
        if jq -e '.i2s.enabled == true and .i2s.device == "hw:APE,0,0"' "$CONFIG_FILE" >/dev/null 2>&1; then
            log_warn "Migrating i2s.device from hw:APE,0,0 to hw:APE,0 (Jetson APE/I2S default)"
            local migrated_path="${CONFIG_FILE}.migrated"
            if jq '(.i2s.device = "hw:APE,0")' "$CONFIG_FILE" > "$migrated_path" 2>/dev/null; then
                mv -f "$migrated_path" "$CONFIG_FILE"
            else
                rm -f "$migrated_path" 2>/dev/null || true
                log_warn "Failed to migrate i2s.device (keeping current config as-is)"
            fi
        fi
    }

    # Optional reset via env
    if [[ "$reset_flag" == "true" || "$reset_flag" == "1" ]]; then
        log_warn "Reset requested via MAGICBOX_RESET_CONFIG, restoring default config"
        cp -f "$DEFAULT_CONFIG" "$CONFIG_FILE"
        apply_profile_on_seed=true
    else
        # Seed if missing/empty
        if [[ ! -s "$CONFIG_FILE" ]]; then
            log_info "Config not found, seeding default config to $CONFIG_FILE"
            cp -f "$DEFAULT_CONFIG" "$CONFIG_FILE"
            apply_profile_on_seed=true
        else
            # Validate existing JSON; if invalid, back up and restore defaults
            if validate_json "$CONFIG_FILE"; then
                log_info "Using existing config at $CONFIG_FILE"
            else
                log_warn "Config is invalid JSON, backing up to $backup_path and restoring default"
                cp -f "$CONFIG_FILE" "$backup_path" || true
                cp -f "$DEFAULT_CONFIG" "$CONFIG_FILE"
                apply_profile_on_seed=true
            fi
        fi
    fi

    # Always merge newly introduced default keys (e.g., i2s) into existing config safely.
    if validate_json "$CONFIG_FILE"; then
        merge_defaults
        if [[ "${apply_profile_on_seed}" == "true" ]]; then
            apply_profile
        fi
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
    # Ensure ALSA is ready (cold boot / device enumeration delay).
    wait_for_alsa

    # If I2S capture is enabled, apply Jetson APE routing in container (best-effort).
    if jq -e '.i2s.enabled == true' "$CONFIG_FILE" >/dev/null 2>&1; then
        configure_jetson_ape_i2s || true
    fi
    exec "$DAEMON"
}

# Start both services (production mode)
start_all() {
    log_info "Starting Magic Box in production mode..."
    log_info "RTP enabled: ${MAGICBOX_ENABLE_RTP}"
    log_info "RTP autostart flag: ${MAGICBOX_RTP_AUTOSTART}"

    # Ensure ALSA is ready (cold boot / device enumeration delay).
    wait_for_alsa

    # If I2S capture is enabled, apply Jetson APE routing in container (best-effort).
    if jq -e '.i2s.enabled == true' "$CONFIG_FILE" >/dev/null 2>&1; then
        configure_jetson_ape_i2s || true
    fi

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
