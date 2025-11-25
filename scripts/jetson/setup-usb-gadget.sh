#!/bin/bash
#
# Magic Box USB Composite Gadget Setup Script
# UAC2 (Audio) + ECM (Ethernet) Composite Device
#
# Usage:
#   ./setup-usb-gadget.sh start   - Configure and enable gadget
#   ./setup-usb-gadget.sh stop    - Disable and cleanup gadget
#   ./setup-usb-gadget.sh restart - Stop and start
#   ./setup-usb-gadget.sh status  - Show current status
#

set -euo pipefail

# === Configuration ===
GADGET_NAME="magicbox"
GADGET_BASE="/sys/kernel/config/usb_gadget"
GADGET_PATH="${GADGET_BASE}/${GADGET_NAME}"

# USB IDs (Linux Foundation VID for development)
USB_VID="0x1d6b"
USB_PID="0x0104"
USB_BCD="0x0100"

# Device strings
MANUFACTURER="MagicBox Audio"
PRODUCT="Magic Box USB Audio"

# === Helper Functions ===

# Generate serial number from device serial
get_serial() {
    local base_serial
    base_serial=$(cat /sys/firmware/devicetree/base/serial-number 2>/dev/null | tr -d '\0' || echo "00000000")
    echo "MB${base_serial: -8}"
}

# Generate consistent MAC address suffix from serial
get_mac_suffix() {
    local serial
    serial=$(get_serial)
    echo "$serial" | md5sum | cut -c1-6
}

# Logging function
log() {
    local level="$1"
    shift
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    logger -t "magicbox-gadget" -p "user.${level}" "$*"
    echo "[$timestamp] [$level] $*"
}

# === Cleanup Function ===
cleanup_gadget() {
    log info "Cleaning up existing gadget configuration..."

    if [[ ! -d "${GADGET_PATH}" ]]; then
        log info "No existing gadget found"
        return 0
    fi

    # Unbind from UDC if bound
    if [[ -f "${GADGET_PATH}/UDC" ]]; then
        local udc
        udc=$(cat "${GADGET_PATH}/UDC" 2>/dev/null || true)
        if [[ -n "$udc" ]]; then
            log info "Unbinding from UDC: $udc"
            echo "" > "${GADGET_PATH}/UDC" 2>/dev/null || true
            sleep 0.5
        fi
    fi

    # Remove configuration symlinks
    for config in "${GADGET_PATH}/configs"/*; do
        [[ -d "$config" ]] || continue
        for link in "$config"/*; do
            if [[ -L "$link" ]]; then
                rm -f "$link"
            fi
        done
        rmdir "$config/strings/0x409" 2>/dev/null || true
        rmdir "$config" 2>/dev/null || true
    done

    # Remove functions
    for func in "${GADGET_PATH}/functions"/*; do
        [[ -d "$func" ]] && rmdir "$func" 2>/dev/null || true
    done

    # Remove strings and gadget directory
    rmdir "${GADGET_PATH}/strings/0x409" 2>/dev/null || true
    rmdir "${GADGET_PATH}" 2>/dev/null || true

    log info "Cleanup complete"
}

# === Setup Function ===
setup_gadget() {
    local serial mac_suffix udc

    log info "Setting up USB Composite Gadget (UAC2 + ECM)..."

    # Ensure configfs is mounted
    if ! mountpoint -q /sys/kernel/config; then
        log info "Mounting configfs..."
        mount -t configfs none /sys/kernel/config
    fi

    # Load required kernel modules
    log info "Loading kernel modules..."
    modprobe libcomposite 2>/dev/null || true
    modprobe usb_f_uac2 2>/dev/null || true
    modprobe usb_f_ecm 2>/dev/null || true

    # Cleanup any existing configuration
    cleanup_gadget

    # Get device-specific identifiers
    serial=$(get_serial)
    mac_suffix=$(get_mac_suffix)

    log info "Device serial: $serial"

    # === Create Gadget ===
    mkdir -p "${GADGET_PATH}"
    cd "${GADGET_PATH}"

    # Set USB device descriptor
    echo "${USB_VID}" > idVendor
    echo "${USB_PID}" > idProduct
    echo "${USB_BCD}" > bcdDevice
    echo 0x0200 > bcdUSB              # USB 2.0
    echo 0xEF > bDeviceClass          # Miscellaneous Device Class
    echo 0x02 > bDeviceSubClass       # Common Class
    echo 0x01 > bDeviceProtocol       # Interface Association Descriptor

    # Set device strings (English)
    mkdir -p strings/0x409
    echo "${serial}" > strings/0x409/serialnumber
    echo "${MANUFACTURER}" > strings/0x409/manufacturer
    echo "${PRODUCT}" > strings/0x409/product

    # === Create UAC2 Function (Audio Input) ===
    log info "Creating UAC2 Audio function..."
    mkdir -p functions/uac2.usb0

    # Playback settings (Host -> Device direction)
    # This appears as "capture" from the gadget's perspective
    echo 3 > functions/uac2.usb0/c_chmask           # Stereo (channels 0 and 1)
    echo "44100,48000" > functions/uac2.usb0/c_srate  # Support both sample rates
    echo 4 > functions/uac2.usb0/c_ssize            # 32-bit samples (4 bytes)

    # Capture settings (Device -> Host direction) - Disabled
    echo 0 > functions/uac2.usb0/p_chmask           # No playback to host

    # === Create ECM Function (Ethernet) ===
    log info "Creating ECM Ethernet function..."
    mkdir -p functions/ecm.usb0

    # Generate consistent MAC addresses based on device serial
    local host_mac="02:${mac_suffix:0:2}:${mac_suffix:2:2}:${mac_suffix:4:2}:00:01"
    local dev_mac="02:${mac_suffix:0:2}:${mac_suffix:2:2}:${mac_suffix:4:2}:00:02"
    echo "$host_mac" > functions/ecm.usb0/host_addr
    echo "$dev_mac" > functions/ecm.usb0/dev_addr

    log info "Host MAC: $host_mac, Device MAC: $dev_mac"

    # === Create Configuration ===
    mkdir -p configs/c.1/strings/0x409
    echo "Magic Box Audio + Network" > configs/c.1/strings/0x409/configuration
    echo 500 > configs/c.1/MaxPower   # 500mA max power draw

    # Link functions to configuration (order matters!)
    # Audio first, then Ethernet
    ln -s functions/uac2.usb0 configs/c.1/
    ln -s functions/ecm.usb0 configs/c.1/

    # === Bind to UDC ===
    udc=$(ls /sys/class/udc/ 2>/dev/null | head -1)
    if [[ -z "$udc" ]]; then
        log error "No USB Device Controller (UDC) found!"
        log error "Check if USB device mode is supported on this port"
        return 1
    fi

    log info "Binding to UDC: $udc"
    echo "$udc" > UDC

    # Verify binding
    sleep 0.5
    if [[ "$(cat UDC 2>/dev/null)" == "$udc" ]]; then
        log info "USB Composite Gadget setup complete!"
        log info "  - UAC2 Audio: 44.1kHz/48kHz, Stereo, 32-bit"
        log info "  - ECM Ethernet: Host=$host_mac, Device=$dev_mac"
        return 0
    else
        log error "Failed to bind to UDC"
        return 1
    fi
}

# === Status Function ===
status_gadget() {
    echo "=== Magic Box USB Gadget Status ==="
    echo

    if [[ ! -d "${GADGET_PATH}" ]]; then
        echo "Status: NOT CONFIGURED"
        echo
        echo "Run '$0 start' to configure the gadget"
        return 1
    fi

    local udc
    udc=$(cat "${GADGET_PATH}/UDC" 2>/dev/null || echo "")

    if [[ -n "$udc" ]]; then
        echo "Status: ACTIVE"
        echo
        echo "UDC: $udc"
        echo "Serial: $(cat ${GADGET_PATH}/strings/0x409/serialnumber 2>/dev/null || echo 'N/A')"
        echo "Product: $(cat ${GADGET_PATH}/strings/0x409/product 2>/dev/null || echo 'N/A')"
        echo
        echo "Functions:"
        for func in "${GADGET_PATH}/functions"/*; do
            if [[ -d "$func" ]]; then
                local func_name
                func_name=$(basename "$func")
                echo "  - $func_name"
                case "$func_name" in
                    uac2.*)
                        echo "      Sample rates: $(cat $func/c_srate 2>/dev/null || echo 'N/A')"
                        echo "      Channels: $(cat $func/c_chmask 2>/dev/null || echo 'N/A')"
                        echo "      Sample size: $(cat $func/c_ssize 2>/dev/null || echo 'N/A') bytes"
                        ;;
                    ecm.*)
                        echo "      Host MAC: $(cat $func/host_addr 2>/dev/null || echo 'N/A')"
                        echo "      Device MAC: $(cat $func/dev_addr 2>/dev/null || echo 'N/A')"
                        ;;
                esac
            fi
        done
        echo

        # Check network interface
        if ip link show usb0 &>/dev/null; then
            echo "Network Interface: usb0"
            ip -4 addr show usb0 2>/dev/null | grep inet | awk '{print "  IP: " $2}'
        fi

        return 0
    else
        echo "Status: CONFIGURED BUT NOT BOUND"
        echo
        echo "The gadget is configured but not connected to a UDC"
        return 1
    fi
}

# === Main Entry Point ===
case "${1:-status}" in
    start)
        setup_gadget
        ;;
    stop)
        cleanup_gadget
        ;;
    restart)
        cleanup_gadget
        sleep 1
        setup_gadget
        ;;
    status)
        status_gadget
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo
        echo "Commands:"
        echo "  start   - Configure and enable USB gadget"
        echo "  stop    - Disable and cleanup USB gadget"
        echo "  restart - Stop and start"
        echo "  status  - Show current gadget status"
        exit 1
        ;;
esac
