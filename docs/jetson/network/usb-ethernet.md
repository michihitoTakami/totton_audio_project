# USB Ethernet 設定

## 概要

Magic BoxはUSB Composite Gadgetの一部としてEthernet機能 (ECM) を提供し、PCからWeb UIへのアクセスを可能にします。

---

## ネットワーク構成

```
┌─────────────────┐                    ┌─────────────────┐
│       PC        │                    │   Magic Box     │
│                 │                    │   (Jetson)      │
│  ┌───────────┐  │   USB Cable       │  ┌───────────┐  │
│  │  Browser  │  │                   │  │  Web UI   │  │
│  │           │  │  ┌─────────────┐  │  │  :80      │  │
│  └─────┬─────┘  │  │             │  │  └─────┬─────┘  │
│        │        │  │   ECM/NCM   │  │        │        │
│  ┌─────┴─────┐  │  │  Ethernet   │  │  ┌─────┴─────┐  │
│  │   usb0    │◄─┼──┤   over      ├──┼─►│   usb0    │  │
│  │ .55.100   │  │  │   USB       │  │  │ .55.1     │  │
│  └───────────┘  │  │             │  │  └───────────┘  │
│                 │  └─────────────┘  │                 │
└─────────────────┘                    └─────────────────┘

      DHCP Client                         DHCP Server
   192.168.55.100/24                    192.168.55.1/24
```

---

## IP アドレス設計

| デバイス | インターフェース | IPアドレス | 備考 |
|---------|----------------|-----------|------|
| Jetson (Magic Box) | usb0 | 192.168.55.1/24 | 固定 |
| PC (Host) | usb0等 | 192.168.55.100/24 | DHCP割り当て |

### サブネット選択理由

`192.168.55.0/24` を選択した理由:
- 一般的な家庭用ルーター (192.168.0.x, 192.168.1.x) と競合しない
- Android USB Tethering (192.168.42.x) と競合しない
- NVIDIA Jetsonデフォルト (192.168.55.x) と同じ（互換性）

---

## systemd-networkd 設定

### /etc/systemd/network/50-usb-ethernet.network

```ini
[Match]
Name=usb0

[Network]
Address=192.168.55.1/24
DHCPServer=yes
IPMasquerade=no
LinkLocalAddressing=no

[DHCPServer]
# DHCPプール設定
PoolOffset=99
PoolSize=10
# DNS（自分自身は提供しない）
EmitDNS=no
EmitRouter=no
# リース時間
DefaultLeaseTimeSec=3600
MaxLeaseTimeSec=7200
```

### 設定の有効化

```bash
# systemd-networkd 有効化
sudo systemctl enable systemd-networkd
sudo systemctl restart systemd-networkd

# 確認
networkctl status usb0
```

---

## ファイアウォール設定

### nftables / iptables

USB Ethernet経由のアクセスのみ許可:

```bash
#!/bin/bash
# /usr/local/bin/magicbox-firewall-setup

# nftables使用
cat > /etc/nftables.conf << 'EOF'
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # Established/Related
        ct state established,related accept

        # Loopback
        iif lo accept

        # USB Ethernet (usb0) からのアクセス許可
        iifname "usb0" tcp dport 80 accept    # Web UI
        iifname "usb0" udp dport 5353 accept  # mDNS
        iifname "usb0" udp dport 67 accept    # DHCP

        # ICMPv4 (ping)
        ip protocol icmp accept

        # それ以外はドロップ（ログ）
        log prefix "nftables drop: " drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
EOF

# 適用
nft -f /etc/nftables.conf
```

### Systemdサービス

```ini
# /etc/systemd/system/magicbox-firewall.service
[Unit]
Description=Magic Box Firewall
After=network-pre.target
Before=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/sbin/nft -f /etc/nftables.conf
ExecReload=/usr/sbin/nft -f /etc/nftables.conf

[Install]
WantedBy=multi-user.target
```

---

## PC側の設定

### 自動設定（推奨）

ほとんどのOSでは、USB接続時に自動的にネットワークインターフェースが作成され、DHCPでIPアドレスが割り当てられます。

**確認コマンド**:

```bash
# Linux
ip addr show | grep -A2 usb

# macOS
ifconfig | grep -A5 "enp\|usb"

# Windows (PowerShell)
Get-NetIPAddress | Where-Object InterfaceAlias -like "*USB*"
```

### 手動設定（DHCPが動作しない場合）

```bash
# Linux
sudo ip addr add 192.168.55.100/24 dev usb0
sudo ip link set usb0 up

# macOS
sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0

# Windows (管理者PowerShell)
New-NetIPAddress -InterfaceAlias "USB Ethernet" -IPAddress 192.168.55.100 -PrefixLength 24
```

---

## 接続確認

### Jetson側

```bash
# インターフェース確認
ip addr show usb0

# DHCPサーバ状態
networkctl status usb0

# リース情報
cat /run/systemd/netif/leases/*
```

### PC側

```bash
# 疎通確認
ping 192.168.55.1

# Web UIアクセス
curl http://192.168.55.1/status
```

---

## トラブルシューティング

### インターフェースが作成されない

```bash
# USB Gadget確認
cat /sys/kernel/config/usb_gadget/magicbox/UDC

# ECM関数確認
ls /sys/kernel/config/usb_gadget/magicbox/functions/ecm.usb0/

# カーネルモジュール確認
lsmod | grep ecm
```

### IPアドレスが割り当てられない

```bash
# systemd-networkd ログ
journalctl -u systemd-networkd

# DHCPサーバ状態
networkctl status usb0

# 手動でIP設定
sudo ip addr add 192.168.55.1/24 dev usb0
```

### PC側でDHCP取得できない

1. USB Gadgetが正しく認識されているか確認
2. PC側のDHCPクライアントが動作しているか確認
3. 手動IP設定を試行

---

## セキュリティ考慮事項

### 物理アクセス前提

USB接続は物理的にケーブルを接続する必要があるため、認証なしでもセキュリティリスクは限定的。

### 将来の拡張（必要に応じて）

- Basic認証 / Digest認証
- HTTPS (自己署名証明書)
- APIトークン認証

---

## 関連ドキュメント

- [mdns-avahi.md](./mdns-avahi.md) - mDNS設定
- [../usb-gadget/composite-gadget.md](../usb-gadget/composite-gadget.md) - USB Gadget設計
- [../systemd/service-design.md](../systemd/service-design.md) - サービス設計
