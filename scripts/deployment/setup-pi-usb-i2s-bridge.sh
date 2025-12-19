#!/usr/bin/env bash
set -euo pipefail

# Install and enable systemd unit to auto-start USB->I2S bridge container on Pi reboot.
#
# Usage:
#   ./scripts/deployment/setup-pi-usb-i2s-bridge.sh
#
# Assumptions:
# - docker engine + compose v2 are installed on the Pi
# - this repository is present on the Pi filesystem

if [[ $EUID -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
UNIT_SRC="${ROOT_DIR}/systemd/raspberry_pi/usb-i2s-bridge.service"
UNIT_DST="/etc/systemd/system/usb-i2s-bridge.service"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "Unit file not found: ${UNIT_SRC}" >&2
  exit 1
fi

echo "Installing usb-i2s-bridge systemd unit..."
echo "  repo:  ${ROOT_DIR}"
echo "  unit:  ${UNIT_DST}"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

# Bake absolute paths to avoid relying on WorkingDirectory being identical across machines.
cat > "${tmp}" <<EOF
[Unit]
Description=Magic Box USB->I2S Bridge (Docker Compose)
Wants=network-online.target docker.service
After=network-online.target docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${ROOT_DIR}
ExecStartPre=/bin/sh -c 'for i in $(seq 1 30); do [ -e /dev/snd ] && exit 0; sleep 1; done; echo "[usb-i2s-bridge] /dev/snd not found" >&2; exit 1'
ExecStart=/usr/bin/docker compose --env-file ${ROOT_DIR}/raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env -f ${ROOT_DIR}/raspberry_pi/usb_i2s_bridge/docker-compose.yml up -d
ExecStop=/usr/bin/docker compose --env-file ${ROOT_DIR}/raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env -f ${ROOT_DIR}/raspberry_pi/usb_i2s_bridge/docker-compose.yml down
TimeoutStartSec=0
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

$SUDO install -m 0644 "${tmp}" "${UNIT_DST}"
$SUDO systemctl daemon-reload
$SUDO systemctl enable usb-i2s-bridge.service
$SUDO systemctl restart usb-i2s-bridge.service

echo ""
echo "Done."
echo "Status:"
$SUDO systemctl --no-pager --full status usb-i2s-bridge.service || true
