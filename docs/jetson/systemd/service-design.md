# Systemd サービス設計

## 概要

Totton Audio Projectは複数のSystemdサービスで構成され、適切な依存関係と起動順序で管理されます。

---

## サービス一覧

| サービス | 説明 | Type |
|---------|------|------|
| `totton-audio-gadget.service` | USB Gadget初期化 | oneshot |
| `gpu-upsampler.service` | 音声処理デーモン | notify |
| `totton-audio-web.service` | Web UI (FastAPI) | simple |

---

## 依存関係図

```
                            multi-user.target
                                   │
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      │                      ▼
    totton-audio-gadget.service        │           (other services)
    [USB Gadget初期化]             │
            │                      │
            │ After/Requires       │
            ▼                      │
    systemd-networkd.service       │
    [IP設定・DHCPサーバ]            │
            │                      │
            │ After                │
            ▼                      │
    avahi-daemon.service           │
    [mDNS: totton-audio.local]         │
            │                      │
            │                      │
            │      ┌───────────────┘
            │      │
            ▼      ▼
    gpu-upsampler.service
    [音声処理デーモン]
            │
            │ After/BindsTo
            ▼
    totton-audio-web.service
    [Web UI]
```

---

## サービス定義

### 1. totton-audio-gadget.service

USB Composite Gadgetの初期化。

```ini
[Unit]
Description=Totton Audio Project USB Composite Gadget
Documentation=https://github.com/michihitoTakami/totton_audio/docs/jetson/usb-gadget/
DefaultDependencies=no
Before=network-pre.target
After=local-fs.target systemd-modules-load.service
Requires=local-fs.target

# UDCが存在する場合のみ起動
ConditionPathExists=/sys/class/udc

[Service]
Type=oneshot
RemainAfterExit=yes

ExecStart=/usr/local/bin/totton-audio-gadget-setup start
ExecStop=/usr/local/bin/totton-audio-gadget-setup stop
ExecReload=/usr/local/bin/totton-audio-gadget-setup restart

# 失敗時の再試行
Restart=on-failure
RestartSec=5
StartLimitInterval=60
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
```

---

### 2. gpu-upsampler.service

メインの音声処理デーモン。

```ini
[Unit]
Description=GPU Audio Upsampler Engine
Documentation=https://github.com/michihitoTakami/totton_audio

# 依存関係
After=totton-audio-gadget.service sound.target
Requires=totton-audio-gadget.service

# ガジェットが停止したらこちらも停止
BindsTo=totton-audio-gadget.service

# ハードウェア条件
ConditionPathExists=/dev/nvidia0

[Service]
Type=notify
NotifyAccess=main

# 実行設定
WorkingDirectory=/opt/totton_audio
ExecStart=/opt/totton_audio/bin/gpu_upsampler_alsa
ExecReload=/bin/kill -HUP $MAINPID

# Watchdog
WatchdogSec=30
WatchdogSignal=SIGABRT

# 停止設定（グレースフルシャットダウン）
TimeoutStopSec=5
KillSignal=SIGTERM
FinalKillSignal=SIGKILL

# 再起動ポリシー
Restart=always
RestartSec=2
StartLimitInterval=300
StartLimitBurst=10

# OOM保護（オーディオは重要）
OOMPolicy=continue
OOMScoreAdjust=-900

# リソース制限
MemoryMax=2G
MemoryHigh=1.5G
CPUWeight=200
IOWeight=200

# リアルタイム優先度
Nice=-10
# LimitRTPRIO は Real-Time Priority の設定（プロトコル名ではない）
LimitRTPRIO=99
LimitMEMLOCK=infinity

# セキュリティ
LockPersonality=yes
NoNewPrivileges=yes
PrivateTmp=yes
ProtectClock=yes
ProtectControlGroups=yes
ProtectHome=yes
ProtectHostname=yes
ProtectKernelLogs=yes
ProtectKernelModules=yes
RestrictNamespaces=yes
RestrictSUIDSGID=yes
SystemCallArchitectures=native

# CUDA/ALSA アクセスに必要な権限
AmbientCapabilities=CAP_SYS_NICE
CapabilityBoundingSet=CAP_SYS_NICE CAP_SYS_RESOURCE

# ロギング
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gpu-upsampler

[Install]
WantedBy=multi-user.target
```

---

### 3. totton-audio-web.service

Web UI (FastAPI/uvicorn)。

```ini
[Unit]
Description=Totton Audio Project Web Control Interface
Documentation=https://github.com/michihitoTakami/totton_audio

# 依存関係
After=network.target gpu-upsampler.service
Requires=gpu-upsampler.service

# デーモンが停止したらWebUIも停止
BindsTo=gpu-upsampler.service
PartOf=gpu-upsampler.service

[Service]
Type=simple

# 実行ユーザー（非root）
User=totton_audio
Group=totton_audio

# 実行設定
WorkingDirectory=/opt/totton_audio
ExecStart=/opt/totton_audio/venv/bin/uvicorn web.main:app \
    --host ${TOTTON_AUDIO_WEB_HOST} \
    --port ${TOTTON_AUDIO_WEB_PORT} \
    --workers 1

# 再起動ポリシー
Restart=always
RestartSec=3

# リソース制限
MemoryMax=512M
CPUWeight=50

# セキュリティ
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/opt/totton_audio/data /tmp

# ロギング
StandardOutput=journal
StandardError=journal
SyslogIdentifier=totton-audio-web

[Install]
WantedBy=multi-user.target
```

---

## 起動シーケンス

### 正常起動

```
t=0s    電源ON
        │
t=5s    systemd起動
        │
t=6s    totton-audio-gadget.service 開始
        │   └─ ConfigFS設定、UDCバインド
        │
t=7s    totton-audio-gadget.service 完了
        │
t=7s    systemd-networkd 設定適用
        │   └─ usb0: 192.168.55.1
        │   └─ DHCPサーバ起動
        │
t=8s    avahi-daemon 起動
        │   └─ totton-audio.local 登録
        │
t=10s   gpu-upsampler.service 開始
        │   ├─ フィルタ係数ロード (GPU)
        │   ├─ ALSA初期化
        │   ├─ ZeroMQ IPC開始
        │   └─ sd_notify(READY=1)
        │
t=25s   gpu-upsampler.service Ready
        │
t=26s   totton-audio-web.service 開始
        │   └─ uvicorn ${TOTTON_AUDIO_WEB_HOST}:${TOTTON_AUDIO_WEB_PORT} 起動
        │
t=28s   Ready (Web UI アクセス可能)
```

### 起動時間目標

| マイルストーン | 目標時間 |
|---------------|---------|
| USB Gadget Ready | < 10秒 |
| 音声処理 Ready | < 30秒 |
| Web UI Ready | < 35秒 |

---

## Watchdog 設計

### gpu-upsampler.service

```ini
[Service]
Type=notify
WatchdogSec=30
```

デーモンは30秒ごとにsystemdへハートビートを送信:

```cpp
// src/alsa_daemon.cpp
#include <systemd/sd-daemon.h>

void watchdog_thread() {
    while (running) {
        if (is_healthy()) {
            sd_notify(0, "WATCHDOG=1");
        }
        sleep(10);  // WatchdogSecの1/3程度
    }
}
```

### Watchdogトリガー条件

| 状態 | 対応 |
|------|------|
| 正常動作 | WATCHDOG=1 送信 |
| 軽微なエラー | WATCHDOG=1 継続、ログ記録 |
| 重大なエラー | WATCHDOG停止 → systemd再起動 |
| ハング | タイムアウト → systemd SIGABRT |

---

## ログ設定

### Journald 設定

`/etc/systemd/journald.conf.d/Totton Audio Project.conf`:

```ini
[Journal]
Storage=persistent
Compress=yes
SystemMaxUse=100M
SystemMaxFileSize=10M
MaxRetentionSec=7day
RateLimitInterval=30s
RateLimitBurst=1000
```

### ログ確認コマンド

```bash
# 全サービスのログ
journalctl -u 'Totton Audio Project*' -u gpu-upsampler

# リアルタイム
journalctl -f -u gpu-upsampler

# 直近のエラー
journalctl -p err -u gpu-upsampler --since "1 hour ago"

# ブート以降
journalctl -b -u totton-audio-gadget
```

---

## 運用コマンド

### 起動・停止

```bash
# 全サービス起動
sudo systemctl start totton-audio-gadget gpu-upsampler totton-audio-web

# 全サービス停止
sudo systemctl stop totton-audio-web gpu-upsampler totton-audio-gadget

# 再起動（設定リロード）
sudo systemctl restart gpu-upsampler
```

### ステータス確認

```bash
# 全サービスステータス
systemctl status totton-audio-gadget gpu-upsampler totton-audio-web

# 依存関係確認
systemctl list-dependencies gpu-upsampler
```

### 自動起動設定

```bash
# 有効化
sudo systemctl enable totton-audio-gadget gpu-upsampler totton-audio-web

# 無効化
sudo systemctl disable totton-audio-web gpu-upsampler totton-audio-gadget
```

---

## トラブルシューティング

### サービスが起動しない

```bash
# 詳細ステータス
systemctl status gpu-upsampler -l

# ジャーナル確認
journalctl -xeu gpu-upsampler

# 依存関係確認
systemctl list-dependencies --all gpu-upsampler
```

### Watchdogタイムアウト

```bash
# コアダンプ確認
coredumpctl list
coredumpctl info gpu_upsampler_alsa

# Watchdog状態
systemctl show gpu-upsampler --property=WatchdogTimestamp
```

---

## 関連ドキュメント

- [watchdog.md](./watchdog.md) - Watchdog詳細設定
- [security-hardening.md](./security-hardening.md) - セキュリティ堅牢化
- [../network/usb-ethernet.md](../network/usb-ethernet.md) - ネットワーク設定
