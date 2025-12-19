#!/usr/bin/env bash
set -euo pipefail

# Allow full override
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

CONFIG_PATH="${USB_I2S_CONFIG_PATH:-/var/lib/usb-i2s-bridge/config.env}"
if [[ -n "${CONFIG_PATH}" && ! -f "${CONFIG_PATH}" ]]; then
  mkdir -p "$(dirname "${CONFIG_PATH}")"
  cp /opt/usb-i2s-bridge/raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env "${CONFIG_PATH}"
fi

CHRT_PRIO=${USB_I2S_CHRT_PRIO:-20}
NICE_LEVEL=${USB_I2S_NICE_LEVEL:--11}
launcher=()

if command -v chrt >/dev/null 2>&1 && [[ "${CHRT_PRIO}" =~ ^[0-9]+$ ]] && (( CHRT_PRIO > 0 )); then
  if chrt -f "${CHRT_PRIO}" true 2>/dev/null; then
    launcher=(chrt -f "${CHRT_PRIO}")
  fi
fi

if [[ ${#launcher[@]} -eq 0 ]] && command -v nice >/dev/null 2>&1 && [[ "${NICE_LEVEL}" =~ ^-?[0-9]+$ ]]; then
  if nice -n "${NICE_LEVEL}" true 2>/dev/null; then
    launcher=(nice -n "${NICE_LEVEL}")
  fi
fi

exec "${launcher[@]}" python3 -m raspberry_pi.usb_i2s_bridge
