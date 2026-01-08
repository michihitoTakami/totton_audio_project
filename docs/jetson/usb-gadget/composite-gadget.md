# USB Composite Gadget 設計

## 概要

Totton Audio Projectは、単一のUSB Type-Cポートで**オーディオ入力**と**管理アクセス**の両方を提供するUSB Composite Gadgetを実装します。

```
┌─────────────────────────────────────────┐
│         USB Composite Device            │
│                                         │
│  ┌─────────────┐   ┌─────────────┐     │
│  │    UAC2     │   │   ECM/NCM   │     │
│  │   (Audio)   │   │  (Ethernet) │     │
│  └─────────────┘   └─────────────┘     │
│                                         │
│  VID: 0x1d6b (Linux Foundation)        │
│  PID: 0x0104 (Composite Gadget)        │
└─────────────────────────────────────────┘
```

---

## 機能仕様

### UAC2 (USB Audio Class 2.0)

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| Direction | Playback (Host→Device) | PCからの入力 |
| Sample Rates | 44100, 48000 Hz | 両方サポート |
| Channels | 2 (Stereo) | L/R |
| Bit Depth | 32-bit | S32_LE |
| Transfer Type | Isochronous | リアルタイム転送 |

### ECM/NCM (USB Ethernet)

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| Type | ECM (推奨) または NCM | Windows互換性ならRNDIS |
| Device IP | 192.168.55.1 | 固定 |
| Host IP | DHCP (192.168.55.100) | systemd-networkd提供 |
| Subnet | /24 | 254ホスト |

---

## ConfigFS 構成

### ディレクトリ構造

```
/sys/kernel/config/usb_gadget/Totton Audio Project/
├── idVendor                 # 0x1d6b
├── idProduct                # 0x0104
├── bcdDevice                # 0x0100
├── bcdUSB                   # 0x0200 (USB 2.0)
├── bDeviceClass             # 0xEF (Miscellaneous)
├── bDeviceSubClass          # 0x02
├── bDeviceProtocol          # 0x01 (IAD)
├── strings/
│   └── 0x409/               # English
│       ├── serialnumber     # MBxxxxxxxx
│       ├── manufacturer     # Totton Audio
│       └── product          # Totton Audio USB Audio
├── functions/
│   ├── uac2.usb0/           # Audio Function
│   │   ├── c_chmask         # 3 (Stereo)
│   │   ├── c_srate          # 44100,48000
│   │   ├── c_ssize          # 4 (32-bit)
│   │   ├── p_chmask         # 0 (No capture)
│   │   └── ...
│   └── ecm.usb0/            # Ethernet Function
│       ├── host_addr        # xx:xx:xx:xx:xx:01
│       ├── dev_addr         # xx:xx:xx:xx:xx:02
│       └── ...
├── configs/
│   └── c.1/
│       ├── MaxPower         # 500
│       ├── strings/
│       │   └── 0x409/
│       │       └── configuration  # "Audio + Network"
│       ├── uac2.usb0 -> ../../functions/uac2.usb0
│       └── ecm.usb0 -> ../../functions/ecm.usb0
└── UDC                      # 3550000.usb (bind)
```

---

## 設定スクリプト

### /usr/local/bin/totton-audio-gadget-setup

```bash
#!/bin/bash
#
# Totton Audio Project USB Composite Gadget Setup
# UAC2 (Audio) + ECM (Ethernet)
#

set -euo pipefail

# === Configuration ===
GADGET_NAME="Totton Audio Project"
GADGET_BASE="/sys/kernel/config/usb_gadget"
GADGET_PATH="${GADGET_BASE}/${GADGET_NAME}"

# USB IDs
USB_VID="0x1d6b"    # Linux Foundation (開発用)
USB_PID="0x0104"    # Composite Gadget
USB_BCD="0x0100"    # Device version 1.0.0

# Strings
MANUFACTURER="Totton Audio"
PRODUCT="Totton Audio USB Audio"

# Generate serial from device serial number
get_serial() {
    local base_serial
    base_serial=$(cat /sys/firmware/devicetree/base/serial-number 2>/dev/null | tr -d '\0' || echo "00000000")
    echo "MB${base_serial: -8}"
}

# Generate consistent MAC addresses
get_mac_suffix() {
    local serial
    serial=$(get_serial)
    echo "$serial" | md5sum | cut -c1-6
}

# Logging
log() {
    local level="$1"
    shift
    logger -t "totton-audio-gadget" -p "user.${level}" "$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*"
}

# === Cleanup ===
cleanup_gadget() {
    log info "Cleaning up existing gadget configuration..."

    if [[ ! -d "${GADGET_PATH}" ]]; then
        log info "No existing gadget found"
        return 0
    fi

    # Unbind UDC
    if [[ -f "${GADGET_PATH}/UDC" ]]; then
        local udc
        udc=$(cat "${GADGET_PATH}/UDC" 2>/dev/null || true)
        if [[ -n "$udc" ]]; then
            log info "Unbinding UDC: $udc"
            echo "" > "${GADGET_PATH}/UDC" 2>/dev/null || true
            sleep 0.5
        fi
    fi

    # Remove configuration links
    for link in "${GADGET_PATH}/configs"/*/; do
        [[ -d "$link" ]] || continue
        for func in "$link"*; do
            [[ -L "$func" ]] && rm -f "$func"
        done
        rmdir "$link/strings/0x409" 2>/dev/null || true
        rmdir "$link" 2>/dev/null || true
    done

    # Remove functions
    for func in "${GADGET_PATH}/functions"/*; do
        [[ -d "$func" ]] && rmdir "$func" 2>/dev/null || true
    done

    # Remove strings and gadget
    rmdir "${GADGET_PATH}/strings/0x409" 2>/dev/null || true
    rmdir "${GADGET_PATH}" 2>/dev/null || true

    log info "Cleanup complete"
}

# === Setup ===
setup_gadget() {
    local serial mac_suffix udc

    log info "Setting up USB Composite Gadget..."

    # Ensure configfs is mounted
    if ! mountpoint -q /sys/kernel/config; then
        log info "Mounting configfs..."
        mount -t configfs none /sys/kernel/config
    fi

    # Clean up any existing configuration
    cleanup_gadget

    # Get device-specific values
    serial=$(get_serial)
    mac_suffix=$(get_mac_suffix)

    log info "Serial: $serial"

    # === Create Gadget ===
    mkdir -p "${GADGET_PATH}"
    cd "${GADGET_PATH}"

    # USB IDs
    echo "${USB_VID}" > idVendor
    echo "${USB_PID}" > idProduct
    echo "${USB_BCD}" > bcdDevice
    echo 0x0200 > bcdUSB           # USB 2.0
    echo 0xEF > bDeviceClass       # Miscellaneous
    echo 0x02 > bDeviceSubClass    # Common Class
    echo 0x01 > bDeviceProtocol    # Interface Association Descriptor

    # Strings
    mkdir -p strings/0x409
    echo "${serial}" > strings/0x409/serialnumber
    echo "${MANUFACTURER}" > strings/0x409/manufacturer
    echo "${PRODUCT}" > strings/0x409/product

    # === UAC2 Function (Audio) ===
    log info "Creating UAC2 Audio function..."
    mkdir -p functions/uac2.usb0

    # Playback settings (Host -> Device, i.e., capture from gadget's perspective)
    echo 3 > functions/uac2.usb0/c_chmask       # Stereo (bit 0 + bit 1)
    echo 44100,48000 > functions/uac2.usb0/c_srate  # Multiple sample rates
    echo 4 > functions/uac2.usb0/c_ssize        # 32-bit samples

    # No capture (Device -> Host)
    echo 0 > functions/uac2.usb0/p_chmask       # Disabled

    # === ECM Function (Ethernet) ===
    log info "Creating ECM Ethernet function..."
    mkdir -p functions/ecm.usb0

    # MAC addresses (consistent based on serial)
    echo "02:${mac_suffix:0:2}:${mac_suffix:2:2}:${mac_suffix:4:2}:00:01" > functions/ecm.usb0/host_addr
    echo "02:${mac_suffix:0:2}:${mac_suffix:2:2}:${mac_suffix:4:2}:00:02" > functions/ecm.usb0/dev_addr

    # === Configuration ===
    mkdir -p configs/c.1/strings/0x409
    echo "Totton Audio + Network" > configs/c.1/strings/0x409/configuration
    echo 500 > configs/c.1/MaxPower   # 500mA

    # Link functions to configuration
    ln -s functions/uac2.usb0 configs/c.1/
    ln -s functions/ecm.usb0 configs/c.1/

    # === Bind to UDC ===
    udc=$(ls /sys/class/udc/ | head -1)
    if [[ -z "$udc" ]]; then
        log error "No UDC (USB Device Controller) found!"
        return 1
    fi

    log info "Binding to UDC: $udc"
    echo "$udc" > UDC

    # Verify
    sleep 0.5
    if [[ "$(cat UDC)" == "$udc" ]]; then
        log info "USB Composite Gadget setup complete"
        return 0
    else
        log error "Failed to bind to UDC"
        return 1
    fi
}

# === Status ===
status_gadget() {
    if [[ ! -d "${GADGET_PATH}" ]]; then
        echo "USB Gadget: Not configured"
        return 1
    fi

    local udc
    udc=$(cat "${GADGET_PATH}/UDC" 2>/dev/null || echo "")

    if [[ -n "$udc" ]]; then
        echo "USB Gadget: Active"
        echo "  UDC: $udc"
        echo "  Serial: $(cat ${GADGET_PATH}/strings/0x409/serialnumber 2>/dev/null)"
        echo "  Functions:"
        for func in "${GADGET_PATH}/functions"/*; do
            [[ -d "$func" ]] && echo "    - $(basename $func)"
        done
        return 0
    else
        echo "USB Gadget: Configured but not bound"
        return 1
    fi
}

# === Main ===
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
        exit 1
        ;;
esac
```

---

## Systemd サービス

### /etc/systemd/system/totton-audio-gadget.service

```ini
[Unit]
Description=Totton Audio Project USB Composite Gadget
Documentation=https://github.com/michihitoTakami/totton_audio
DefaultDependencies=no
Before=network-pre.target
After=local-fs.target systemd-modules-load.service
Requires=local-fs.target
ConditionPathExists=/sys/class/udc

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/totton-audio-gadget-setup start
ExecStop=/usr/local/bin/totton-audio-gadget-setup stop
ExecReload=/usr/local/bin/totton-audio-gadget-setup restart

# Recovery
Restart=on-failure
RestartSec=5
StartLimitInterval=60
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
```

---

## VID/PID について

### 開発フェーズ

| 項目 | 値 | 備考 |
|------|-----|------|
| VID | 0x1d6b | Linux Foundation |
| PID | 0x0104 | Multifunction Composite Gadget |

これは開発・プロトタイプ用です。製品版では専用のVID/PID取得を推奨します。

### 製品化時

製品として販売する場合、以下のいずれかが必要です：

1. **USB-IF からVID取得**: $6,000 (2024年時点)
2. **pid.codes 利用**: オープンソース向け無料VID共有
3. **OEM契約**: 既存メーカーのVID下でPID割り当て

---

## Windows互換性

Windows環境での認識を確保するための注意点：

### UAC2

- Windows 10 1703以降: ネイティブUAC2サポート
- それ以前: サードパーティドライバが必要

### Ethernet

| プロトコル | Windows対応 | 備考 |
|-----------|-------------|------|
| ECM | 要ドライバ | Linux/macOSは標準対応 |
| NCM | 要ドライバ | 高速だがドライバ必要 |
| RNDIS | 標準対応 | Windowsネイティブ |

**推奨**: Windows互換性が重要な場合は`rndis.usb0`を使用

```bash
# ECM の代わりに RNDIS を使用
mkdir -p functions/rndis.usb0
ln -s functions/rndis.usb0 configs/c.1/
```

---

## トラブルシューティング

### ガジェットが認識されない

```bash
# UDC の確認
ls /sys/class/udc/

# カーネルモジュール確認
lsmod | grep -E 'libcomposite|usb_f_uac2|usb_f_ecm'

# ログ確認
journalctl -u totton-audio-gadget.service
dmesg | grep -i gadget
```

### オーディオデバイスが見えない

```bash
# ALSA デバイス確認
aplay -l
arecord -l

# UAC2 function 確認
cat /sys/kernel/config/usb_gadget/Totton Audio Project/functions/uac2.usb0/c_srate
```

---

## 関連ドキュメント

- [kernel-modules.md](./kernel-modules.md) - カーネルモジュールビルド
- [configfs-setup.md](./configfs-setup.md) - ConfigFS詳細設定
- [known-issues.md](./known-issues.md) - 既知の制限と代替案
