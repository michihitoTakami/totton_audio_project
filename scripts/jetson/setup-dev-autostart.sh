#!/usr/bin/env bash
# =============================================================================
# Jetson Development Autostart Helper
# -----------------------------------------------------------------------------
# このスクリプトは「Jetson上でGitHubからcloneした開発用ワークツリー」を対象に、
# 再起動後もgpu_upsamplerデーモンとWeb UIを自動で起動させるための
# systemdサービス(gpu-upsampler-dev / magicbox-web-dev)を作成・削除します。
#
# ⚠️ Production用(正規バイナリ / /opt/magicbox レイアウト)では使用しないでください。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"

UPS_SERVICE_NAME="gpu-upsampler-dev.service"
WEB_SERVICE_NAME="magicbox-web-dev.service"
UPS_SERVICE_PATH="/etc/systemd/system/$UPS_SERVICE_NAME"
WEB_SERVICE_PATH="/etc/systemd/system/$WEB_SERVICE_NAME"

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
    ACTION="install"
else
    shift
fi

REPO_ROOT="$REPO_ROOT_DEFAULT"
DEV_USER="${MAGICBOX_DEV_USER:-jetson}"
WEB_PORT="${MAGICBOX_DEV_PORT:-80}"

usage() {
    cat <<'EOF'
Usage:
  sudo scripts/jetson/setup-dev-autostart.sh [install|uninstall|status] [options]

Options:
  --repo <path>   : 任意のワークツリーパス（デフォルト: スクリプト位置から推定）
  --user <name>   : Web UIを実行するユーザー（デフォルト: $MAGICBOX_DEV_USER または "jetson"）
  --port <port>   : Web UIの待受ポート（デフォルト: $MAGICBOX_DEV_PORT または 80）

環境変数:
  MAGICBOX_DEV_USER : --user のデフォルト値
  MAGICBOX_DEV_PORT : --port のデフォルト値
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            [[ $# -lt 2 ]] && { echo "[ERROR] --repo requires a path" >&2; exit 1; }
            REPO_ROOT="$(realpath "$2")"
            shift 2
            ;;
        --user)
            [[ $# -lt 2 ]] && { echo "[ERROR] --user requires a value" >&2; exit 1; }
            DEV_USER="$2"
            shift 2
            ;;
        --port)
            [[ $# -lt 2 ]] && { echo "[ERROR] --port requires a value" >&2; exit 1; }
            WEB_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

info() {
    echo -e "[INFO] $*"
}

warn() {
    echo -e "[WARN] $*" >&2
}

error() {
    echo -e "[ERROR] $*" >&2
    exit 1
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        error "root権限が必要です。sudo経由で実行してください。"
    fi
}

ensure_binary_exists() {
    local path="$1"
    if [[ ! -x "$path" ]]; then
        warn "バイナリが見つかりません: $path"
        warn "Jetson上で 'cmake -B build && cmake --build build -j\$(nproc)' を先に実行してください。"
    fi
}

detect_uv_bin() {
    local uv_path
    uv_path="$(command -v uv || true)"
    [[ -n "$uv_path" ]] || error "'uv' コマンドが見つかりません。https://github.com/astral-sh/uv をインストールしてください。"
    echo "$uv_path"
}

ensure_user_exists() {
    if ! id "$DEV_USER" >/dev/null 2>&1; then
        error "ユーザー '$DEV_USER' が存在しません。--user で既存ユーザーを指定してください。"
    fi
}

write_upsampler_service() {
    local binary="$1"
    cat > "$UPS_SERVICE_PATH" <<EOF
[Unit]
Description=GPU Upsampler (dev worktree: $REPO_ROOT)
After=network.target sound.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
ExecStart=$binary
Restart=always
RestartSec=2
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
LimitRTPRIO=99
LimitMEMLOCK=infinity
Nice=-10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

write_web_service() {
    local uv_bin="$1"
    local cap_section=""
    if [[ "$WEB_PORT" -lt 1024 ]]; then
        cap_section=$'AmbientCapabilities=CAP_NET_BIND_SERVICE\nCapabilityBoundingSet=CAP_NET_BIND_SERVICE'
    fi

    local dev_group
    dev_group="$(id -gn "$DEV_USER")"

    cat > "$WEB_SERVICE_PATH" <<EOF
[Unit]
Description=Magic Box Web UI (dev worktree: $REPO_ROOT)
After=network.target gpu-upsampler-dev.service
Requires=gpu-upsampler-dev.service
PartOf=gpu-upsampler-dev.service

[Service]
Type=simple
User=$DEV_USER
Group=$dev_group
WorkingDirectory=$REPO_ROOT
ExecStart=$uv_bin run uvicorn web.main:app --host 0.0.0.0 --port $WEB_PORT --workers 1 --log-level warning
Restart=always
RestartSec=2
Environment=UV_LINK_MODE=copy
$cap_section
NoNewPrivileges=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

install_services() {
    require_root
    ensure_user_exists

    local repo_binary="$REPO_ROOT/build/gpu_upsampler_alsa"
    ensure_binary_exists "$repo_binary"

    local uv_bin
    uv_bin="$(detect_uv_bin)"

    info "開発用ワークツリー: $REPO_ROOT"
    info "Web UI実行ユーザー: $DEV_USER"
    info "Web UIポート     : $WEB_PORT"
    info "uvコマンド       : $uv_bin"

    write_upsampler_service "$repo_binary"
    info "Created $UPS_SERVICE_PATH"

    write_web_service "$uv_bin"
    info "Created $WEB_SERVICE_PATH"

    systemctl daemon-reload
    systemctl enable --now "$UPS_SERVICE_NAME"
    systemctl enable --now "$WEB_SERVICE_NAME"

    info "自動起動のセットアップが完了しました。"
    info "ステータス確認: sudo systemctl status $UPS_SERVICE_NAME $WEB_SERVICE_NAME"
}

uninstall_services() {
    require_root

    systemctl disable --now "$WEB_SERVICE_NAME" >/dev/null 2>&1 || true
    systemctl disable --now "$UPS_SERVICE_NAME" >/dev/null 2>&1 || true

    rm -f "$WEB_SERVICE_PATH" "$UPS_SERVICE_PATH"
    systemctl daemon-reload

    info "自動起動設定を削除しました。"
}

show_status() {
    systemctl status "$UPS_SERVICE_NAME" "$WEB_SERVICE_NAME"
}

case "$ACTION" in
    install)
        install_services
        ;;
    uninstall|remove|disable)
        uninstall_services
        ;;
    status)
        show_status
        ;;
    *)
        echo "[ERROR] 不正なアクション: $ACTION" >&2
        usage
        exit 1
        ;;
esac

