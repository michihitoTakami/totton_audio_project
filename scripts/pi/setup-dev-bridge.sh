#!/usr/bin/env bash
#
# Raspberry Pi bridge bootstrapper for Magic Box
# -------------------------------------------------------------
# - Presents the Pi as UAC2 + ECM composite device to the host PC
# - Forwards the incoming PCM stream to Jetson via PipeWire RTP
# - Configures static networking for usb0 (PC link) and Jetson link
# - Sets up nginx reverse proxy to Jetson's Web UI (:8000)
# -------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
    ACTION="install"
else
    shift
fi

# Defaults (override via CLI flags)
PI_USER="${MAGICBOX_PI_USER:-pi}"
PC_SUBNET="${MAGICBOX_PC_SUBNET:-192.168.55.1/24}"
JETSON_IF="${MAGICBOX_JETSON_IF:-eth1}"
JETSON_IP="${MAGICBOX_JETSON_IP:-192.168.56.2}"
PI_JETSON_IP="${MAGICBOX_PI_JETSON_IP:-192.168.56.1/24}"
RTP_PORT="${MAGICBOX_RTP_PORT:-6000}"
RTP_RATE="${MAGICBOX_RTP_RATE:-768000}"
NGINX_LISTEN="${MAGICBOX_NGINX_LISTEN:-80}"
JETSON_WEB_PORT="${MAGICBOX_JETSON_WEB_PORT:-8000}"
USB_GADGET_SAMPLE_RATES="${MAGICBOX_SAMPLE_RATES:-44100,48000,88200,96000,176400,192000,352800,384000,705600,768000}"

usage() {
    cat <<EOF
Usage:
  sudo scripts/pi/setup-dev-bridge.sh [install|uninstall|status] [options]

Options:
  --pi-user <user>       : Linux user that runs PipeWire/nginx proxy (default: $PI_USER)
  --pc-subnet <cidr>     : Static IP for usb0 towards PC (default: $PC_SUBNET)
  --jetson-if <ifname>   : Network interface that connects to Jetson (default: $JETSON_IF)
  --pi-jetson-ip <cidr>  : Static IP for Pi on Jetson link (default: $PI_JETSON_IP)
  --jetson-ip <ip>       : Jetson address reachable from Pi (default: $JETSON_IP)
  --rtp-port <port>      : UDP port for RTP stream (default: $RTP_PORT)
  --rtp-rate <hz>        : Sample rate announced to RTP sink (default: $RTP_RATE)
  --nginx-listen <port>  : Port exposed on Pi for Web UI proxy (default: $NGINX_LISTEN)
  --jetson-web-port <p>  : Jetson FastAPI port to proxy (default: $JETSON_WEB_PORT)
  --sample-rates <csv>   : Comma separated rates advertised to UAC2 host
  -h, --help             : Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pi-user)
            PI_USER="$2"
            shift 2
            ;;
        --pc-subnet)
            PC_SUBNET="$2"
            shift 2
            ;;
        --jetson-if)
            JETSON_IF="$2"
            shift 2
            ;;
        --pi-jetson-ip)
            PI_JETSON_IP="$2"
            shift 2
            ;;
        --jetson-ip)
            JETSON_IP="$2"
            shift 2
            ;;
        --rtp-port)
            RTP_PORT="$2"
            shift 2
            ;;
        --rtp-rate)
            RTP_RATE="$2"
            shift 2
            ;;
        --nginx-listen)
            NGINX_LISTEN="$2"
            shift 2
            ;;
        --jetson-web-port)
            JETSON_WEB_PORT="$2"
            shift 2
            ;;
        --sample-rates)
            USB_GADGET_SAMPLE_RATES="$2"
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

PI_UID=""
PI_HOME=""

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "[ERROR] root privileges are required." >&2
        exit 1
    fi
}

detect_user() {
    if ! id "$PI_USER" >/dev/null 2>&1; then
        echo "[ERROR] User '$PI_USER' does not exist on this system." >&2
        exit 1
    fi
    PI_UID=$(id -u "$PI_USER")
    PI_HOME=$(getent passwd "$PI_USER" | cut -d: -f6)
}

install_packages() {
    echo "[INFO] Installing required packages..."
    apt-get update
    apt-get install -y \
        pipewire pipewire-bin pipewire-audio-client-libraries pipewire-pulse pipewire-alsa \
        jq nginx
}

install_gadget_artifacts() {
    local gadget_script="/usr/local/bin/magicbox-pi-gadget"
    cat > "$gadget_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

GADGET_NAME="magicbox"
GADGET_BASE="/sys/kernel/config/usb_gadget"
GADGET_PATH="${GADGET_BASE}/${GADGET_NAME}"
USB_VID="0x1d6b"
USB_PID="0x0104"
USB_BCD="0x0100"
MANUFACTURER="MagicBox Audio"
PRODUCT="Magic Box Pi Bridge"
SAMPLE_RATES="${MAGICBOX_SAMPLE_RATES:-44100,48000,88200,96000,176400,192000,352800,384000,705600,768000}"

get_serial() {
    local base_serial
    base_serial=$(cat /sys/firmware/devicetree/base/serial-number 2>/dev/null | tr -d '\0' || echo "00000000")
    echo "MBPI${base_serial: -8}"
}

get_mac_suffix() {
    local serial
    serial=$(get_serial)
    echo "$serial" | md5sum | cut -c1-6
}

log() { echo "[magicbox-pi-gadget] $*"; }

cleanup() {
    [[ -d "$GADGET_PATH" ]] || return 0
    if [[ -f "$GADGET_PATH/UDC" ]]; then
        local udc
        udc=$(cat "$GADGET_PATH/UDC" 2>/dev/null || true)
        if [[ -n "$udc" ]]; then
            echo "" > "$GADGET_PATH/UDC"
            sleep 0.2
        fi
    fi
    find "$GADGET_PATH/configs" -maxdepth 1 -mindepth 1 -type d -exec bash -c 'rm -f "$0"/*' {} \; 2>/dev/null || true
    find "$GADGET_PATH/functions" -maxdepth 1 -mindepth 1 -type d -exec rmdir {} \; 2>/dev/null || true
    rm -rf "$GADGET_PATH/strings"
    rmdir "$GADGET_PATH" 2>/dev/null || true
}

setup() {
    modprobe libcomposite
    modprobe usb_f_uac2
    modprobe usb_f_ecm
    mount -t configfs none /sys/kernel/config 2>/dev/null || true

    cleanup

    local serial mac_suffix host_mac dev_mac
    serial=$(get_serial)
    mac_suffix=$(get_mac_suffix)
    host_mac="02:${mac_suffix:0:2}:${mac_suffix:2:2}:${mac_suffix:4:2}:00:01"
    dev_mac="02:${mac_suffix:0:2}:${mac_suffix:2:2}:${mac_suffix:4:2}:00:02"

    mkdir -p "$GADGET_PATH"
    cd "$GADGET_PATH"
    echo "$USB_VID" > idVendor
    echo "$USB_PID" > idProduct
    echo "$USB_BCD" > bcdDevice
    echo 0x0200 > bcdUSB
    echo 0xEF > bDeviceClass
    echo 0x02 > bDeviceSubClass
    echo 0x01 > bDeviceProtocol

    mkdir -p strings/0x409
    echo "$serial" > strings/0x409/serialnumber
    echo "$MANUFACTURER" > strings/0x409/manufacturer
    echo "$PRODUCT" > strings/0x409/product

    mkdir -p functions/uac2.usb0
    echo 3 > functions/uac2.usb0/c_chmask
    echo "$SAMPLE_RATES" > functions/uac2.usb0/c_srate
    echo 4 > functions/uac2.usb0/c_ssize
    echo 0 > functions/uac2.usb0/p_chmask

    mkdir -p functions/ecm.usb0
    echo "$host_mac" > functions/ecm.usb0/host_addr
    echo "$dev_mac" > functions/ecm.usb0/dev_addr

    mkdir -p configs/c.1/strings/0x409
    echo "Magic Box Pi Bridge" > configs/c.1/strings/0x409/configuration
    echo 500 > configs/c.1/MaxPower
    ln -s functions/uac2.usb0 configs/c.1/
    ln -s functions/ecm.usb0 configs/c.1/

    local udc
    udc=$(ls /sys/class/udc/ | head -1)
    if [[ -z "$udc" ]]; then
        log "No UDC available"
        return 1
    fi
    echo "$udc" > UDC
    log "USB gadget ready (UDC=$udc, serial=$serial)"
}

case "${1:-start}" in
    start) setup ;;
    stop) cleanup ;;
    restart) cleanup; sleep 1; setup ;;
    status)
        if [[ -d "$GADGET_PATH" ]]; then
            cat <<INFO
Gadget: $GADGET_PATH
UDC: $(cat "$GADGET_PATH/UDC" 2>/dev/null || echo "N/A")
Functions: $(ls "$GADGET_PATH/functions" 2>/dev/null | tr '\n' ' ')
INFO
        else
            echo "Gadget not configured"
        fi
        ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac
EOF
    chmod +x "$gadget_script"

    cat > /etc/systemd/system/magicbox-pi-gadget.service <<EOF
[Unit]
Description=Magic Box Pi USB Gadget (UAC2 + ECM)
After=local-fs.target sysinit.target
ConditionPathExists=/sys/kernel/config

[Service]
Type=oneshot
RemainAfterExit=yes
Environment=MAGICBOX_SAMPLE_RATES=${USB_GADGET_SAMPLE_RATES}
ExecStart=${gadget_script} start
ExecStop=${gadget_script} stop
ExecReload=${gadget_script} restart

[Install]
WantedBy=multi-user.target
EOF
}

configure_dhcpcd() {
    local marker_start="# >>> MAGICBOX PI BRIDGE"
    local marker_end="# <<< MAGICBOX PI BRIDGE"
    local conf="/etc/dhcpcd.conf"
    sed -i "/${marker_start}/,/${marker_end}/d" "$conf" 2>/dev/null || true
    cat >> "$conf" <<EOF
${marker_start}
interface usb0
static ip_address=${PC_SUBNET}
nogateway

interface ${JETSON_IF}
static ip_address=${PI_JETSON_IP}
nogateway
${marker_end}
EOF
}

configure_boot_overlay() {
    local config_path=""
    for candidate in /boot/firmware/config.txt /boot/config.txt; do
        if [[ -f "$candidate" ]]; then
            config_path="$candidate"
            break
        fi
    done

    if [[ -z "$config_path" ]]; then
        echo "[WARN] Could not locate config.txt to configure dwc2 overlay." >&2
        return
    fi

    if ! grep -q "dtoverlay=dwc2" "$config_path"; then
        echo "dtoverlay=dwc2,dr_mode=peripheral" >> "$config_path"
        echo "[INFO] Added dwc2 overlay to $config_path"
    fi

    if ! grep -q "^dwc2" /etc/modules 2>/dev/null; then
        echo "dwc2" >> /etc/modules
    fi
    if ! grep -q "^libcomposite" /etc/modules 2>/dev/null; then
        echo "libcomposite" >> /etc/modules
    fi
}

install_rtp_forwarder() {
    local forward_script="/usr/local/bin/magicbox-rtp-forward.sh"
    cat > "$forward_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

JETSON_IP="${JETSON_IP:-192.168.56.2}"
RTP_PORT="${RTP_PORT:-6000}"
RTP_RATE="${RTP_RATE:-768000}"
JETSON_IF="${JETSON_IF:-eth1}"
CAPTURE_PATTERN="${CAPTURE_PATTERN:-alsa_input.usb-Magic_Box}"

wait_pipewire() {
    for _ in {1..30}; do
        if pw-cli info 0 >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "[magicbox-rtp] PipeWire not ready" >&2
    return 1
}

get_node_id() {
    local pattern="$1"
    pw-dump | jq -r ".[] | select(.type==\"PipeWire:Interface:Node\" and (.info.props.\"node.name\" | test(\"${pattern}\"))) | .id" | head -n1
}

load_rtp_sink() {
    if pw-dump | jq -r '.[] | select(.info.props."node.name"=="magicbox-rtp-sink") | .id' | grep -q .; then
        return 0
    fi
    pw-cli load-module libpipewire-module-rtp-sink "node.name=magicbox-rtp-sink destination.ip=${JETSON_IP} destination.port=${RTP_PORT} local.ifname=${JETSON_IF} mtu=1024 stream.props='media.class=Audio/Sink;audio.format=F32;audio.rate=${RTP_RATE};audio.channels=2;node.description=MagicBox RTP Sink'"
}

start_loopback() {
    local capture_id sink_id
    capture_id=$(get_node_id "$CAPTURE_PATTERN")
    if [[ -z "$capture_id" ]]; then
        echo "[magicbox-rtp] Capture node not found (pattern=${CAPTURE_PATTERN})" >&2
        return 1
    fi
    sink_id=$(get_node_id "magicbox-rtp-sink")
    if [[ -z "$sink_id" ]]; then
        echo "[magicbox-rtp] RTP sink node not found" >&2
        return 1
    fi
    exec pw-loopback --capture-props="node.name=magicbox-loop-capture target.object=${capture_id}" \
        --playback-props="node.name=magicbox-loop-rtp target.object=${sink_id}" \
        --latency=256/${RTP_RATE}
}

wait_pipewire
load_rtp_sink
sleep 1
start_loopback
EOF
    chmod +x "$forward_script"

    local pi_group
    pi_group=$(id -gn "$PI_USER")

    cat > /etc/systemd/system/magicbox-rtp-forward.service <<EOF
[Unit]
Description=Magic Box RTP forwarder
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${PI_USER}
Group=${pi_group}
Environment=XDG_RUNTIME_DIR=/run/user/${PI_UID}
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/${PI_UID}/bus
Environment=JETSON_IP=${JETSON_IP}
Environment=RTP_PORT=${RTP_PORT}
Environment=RTP_RATE=${RTP_RATE}
Environment=JETSON_IF=${JETSON_IF}
Environment=CAPTURE_PATTERN=alsa_input.usb-Magic_Box
ExecStart=${forward_script}
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    loginctl enable-linger "$PI_USER" >/dev/null 2>&1 || true
}

configure_nginx() {
    local site="/etc/nginx/sites-available/magicbox"
    cat > "$site" <<EOF
server {
    listen ${NGINX_LISTEN};
    server_name magicbox-pi;

    location / {
        proxy_pass http://${JETSON_IP}:${JETSON_WEB_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    ln -sf "$site" /etc/nginx/sites-enabled/magicbox
    rm -f /etc/nginx/sites-enabled/default
    systemctl enable --now nginx
    systemctl reload nginx
}

enable_services() {
    systemctl daemon-reload
    systemctl enable --now magicbox-pi-gadget.service
    systemctl enable --now magicbox-rtp-forward.service
}

uninstall_all() {
    systemctl disable --now magicbox-pi-gadget.service >/dev/null 2>&1 || true
    systemctl disable --now magicbox-rtp-forward.service >/dev/null 2>&1 || true
    rm -f /etc/systemd/system/magicbox-pi-gadget.service
    rm -f /etc/systemd/system/magicbox-rtp-forward.service
    rm -f /usr/local/bin/magicbox-pi-gadget
    systemctl daemon-reload

    sed -i "/# >>> MAGICBOX PI BRIDGE/,/# <<< MAGICBOX PI BRIDGE/d" /etc/dhcpcd.conf 2>/dev/null || true

    rm -f /usr/local/bin/magicbox-rtp-forward.sh

    rm -f /etc/nginx/sites-enabled/magicbox /etc/nginx/sites-available/magicbox
    systemctl reload nginx >/dev/null 2>&1 || true
}

show_status() {
    echo "=== Magic Box Pi Bridge Status ==="
    systemctl status magicbox-pi-gadget.service --no-pager || true
    systemctl status magicbox-rtp-forward.service --no-pager || true
    systemctl status nginx --no-pager || true
    echo ""
    echo "USB gadget:"
    /usr/local/bin/magicbox-pi-gadget status || true
}

main() {
    require_root
    detect_user

    case "$ACTION" in
        install)
            install_packages
            install_gadget_artifacts
            configure_dhcpcd
            configure_boot_overlay
            install_rtp_forwarder
            configure_nginx
            enable_services
            echo "[INFO] Installation complete. Reboot Pi to ensure usb0 enumerates."
            ;;
        uninstall)
            uninstall_all
            echo "[INFO] Pi bridge configuration removed."
            ;;
        status)
            show_status
            ;;
        *)
            echo "[ERROR] Unknown action: $ACTION" >&2
            usage
            exit 1
            ;;
    esac
}

main

