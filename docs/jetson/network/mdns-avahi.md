# mDNS (Avahi) 設定

## 概要

Magic Boxは mDNS (multicast DNS) を使用して、`magicbox.local` としてネットワーク上で自動的に発見可能になります。これにより、IPアドレスを覚えなくてもブラウザからアクセスできます。

---

## 機能

| 機能 | 説明 |
|------|------|
| ホスト名解決 | `magicbox.local` → `192.168.55.1` |
| サービス広告 | `_http._tcp` (Web UI) |
| ゼロコンフィグ | PC側の追加設定不要 |

---

## Avahi 設定

### /etc/avahi/avahi-daemon.conf

```ini
[server]
# ホスト名（.local は自動付加）
host-name=magicbox

# ドメイン
domain-name=local

# 使用プロトコル
use-ipv4=yes
use-ipv6=no

# インターフェース制限（USB Ethernetのみ）
allow-interfaces=usb0
deny-interfaces=eth0,wlan0

# リフレクター無効（単一ネットワーク）
enable-reflector=no

# DBus許可
enable-dbus=yes

[publish]
# 自分自身を公開
publish-addresses=yes
publish-hinfo=yes
publish-workstation=yes

# ユーザーサービス公開許可
disable-user-service-publishing=no

[reflector]
enable-reflector=no

[rlimits]
rlimit-core=0
rlimit-data=4194304
rlimit-fsize=0
rlimit-nofile=768
rlimit-stack=4194304
rlimit-nproc=3
```

---

## サービス広告

### /etc/avahi/services/magicbox-http.service

```xml
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
    <name replace-wildcards="yes">Magic Box Web UI on %h</name>

    <service>
        <type>_http._tcp</type>
        <port>80</port>
        <txt-record>path=/</txt-record>
        <txt-record>version=1.0</txt-record>
    </service>
</service-group>
```

### 広告内容

```
Service: _http._tcp
Name: Magic Box Web UI on magicbox
Host: magicbox.local
Port: 80
TXT: path=/, version=1.0
```

---

## 有効化

```bash
# Avahi インストール（通常はプリインストール）
sudo apt install avahi-daemon

# サービスファイル配置
sudo cp magicbox-http.service /etc/avahi/services/

# 再起動
sudo systemctl restart avahi-daemon

# 自動起動有効化
sudo systemctl enable avahi-daemon
```

---

## 動作確認

### Jetson側

```bash
# Avahiステータス
sudo systemctl status avahi-daemon

# 公開中のサービス確認
avahi-browse -all -r

# 自分自身の解決テスト
avahi-resolve -n magicbox.local
```

### PC側 (Linux)

```bash
# mDNS解決
avahi-resolve -n magicbox.local

# または
ping magicbox.local

# サービス検索
avahi-browse _http._tcp
```

### PC側 (macOS)

```bash
# mDNS解決（Bonjourネイティブ対応）
ping magicbox.local

# サービス検索
dns-sd -B _http._tcp

# 詳細
dns-sd -L "Magic Box Web UI on magicbox" _http._tcp local.
```

### PC側 (Windows)

```powershell
# Windows 10以降はmDNS対応
ping magicbox.local

# または Bonjour Print Services インストール後
dns-sd -B _http._tcp
```

---

## プラットフォーム別対応状況

| OS | mDNS対応 | 備考 |
|----|---------|------|
| Linux | ✓ | avahi-daemon / systemd-resolved |
| macOS | ✓ | Bonjour (ネイティブ) |
| Windows 10+ | ✓ | ネイティブ対応 |
| Windows 7/8 | △ | Bonjour インストール必要 |
| iOS | ✓ | Bonjour (ネイティブ) |
| Android | △ | アプリ依存 |

---

## ホスト名の競合

同一ネットワーク上に複数のMagic Boxがある場合:

```
magicbox.local        (最初のデバイス)
magicbox-2.local      (2台目)
magicbox-3.local      (3台目)
```

Avahiが自動的に番号を付加して競合を回避します。

---

## カスタムホスト名

デバイスごとにユニークな名前を設定する場合:

```bash
# ホスト名変更
sudo hostnamectl set-hostname magicbox-living

# Avahi設定更新
sudo systemctl restart avahi-daemon
```

アクセス: `http://magicbox-living.local/`

---

## トラブルシューティング

### magicbox.local が解決できない

```bash
# Avahi動作確認
sudo systemctl status avahi-daemon
journalctl -u avahi-daemon

# インターフェース確認
avahi-daemon --check

# mDNSパケット確認（Jetson側）
sudo tcpdump -i usb0 port 5353
```

### 名前が競合している

```bash
# 現在の名前確認
avahi-resolve -n magicbox.local

# ホスト名変更
sudo hostnamectl set-hostname magicbox-unique
```

### Windows で認識されない

1. Bonjourサービス確認:
   ```powershell
   Get-Service -Name "Bonjour Service"
   ```

2. ファイアウォール確認:
   - UDP 5353 (mDNS) を許可

3. 代替手段:
   - IPアドレス直接指定 (`http://192.168.55.1/`)

---

## セキュリティ

### mDNS のリスク

- ネットワーク上にデバイス名を広告
- USB Ethernet (ポイントツーポイント) なので影響限定的

### 無効化（必要な場合）

```bash
# Avahi無効化
sudo systemctl stop avahi-daemon
sudo systemctl disable avahi-daemon
```

IPアドレス直接アクセス (`192.168.55.1`) は引き続き可能。

---

## 関連ドキュメント

- [usb-ethernet.md](./usb-ethernet.md) - USB Ethernet設定
- [../systemd/service-design.md](../systemd/service-design.md) - サービス設計
