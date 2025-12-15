# Jetson: snd-aloop（ALSA Loopback）セットアップガイド

Jetson Orin Nano での `snd-aloop` モジュール自動ロード手順と systemd統合ガイド。

## 背景

### なぜ snd-aloop が必要か

Magic Box の データパイプライン：

```
RTP受信 (GStreamer)
  ↓
ALSA Loopback (仮想デバイス)
  ↓
GPU Convolution エンジン
  ↓
USB DAC
```

- `jetson-pcm-receiver` の出力先は **ALSA Loopback** (`hw:Loopback,0,0`)
- Jetsonホストで `snd-aloop` カーネルモジュールをロードする必要がある
- コンテナ内では `modprobe` が実行できないため、**ホスト側での事前ロードが必須**

## 前提条件

- Jetson Orin Nano Super（または互換デバイス）
- Linux カーネル 5.15+ (標準搭載)
- `root` または `sudo` 権限

## 手動ロード（確認用）

### 1. モジュール確認

```bash
# 現在ロードされているサウンドモジュール一覧
lsmod | grep snd

# 出力例（ロード済み時）:
# snd_aloop              20480  0
# snd_pcm               143360  4 snd_aloop,snd_hda_intel,...
# snd                   118784  18 snd_aloop,snd_hda_intel,...
```

### 2. モジュール手動ロード

```bash
# snd-aloop をロード
sudo modprobe snd_aloop

# ロード確認
lsmod | grep snd_aloop

# ALSA が認識しているか確認
arecord -l          # Loopback デバイスが表示されるか確認
aplay -l            # 出力側も確認
```

### 3. ロード時の確認

```bash
# hw:Loopback が見つかるか
arecord -l | grep Loopback
aplay -l | grep Loopback

# hw:Loopback,0,0 として使用可能か確認（例）
timeout 1 arecord -D hw:Loopback,0,0 -f S16_LE -r 48000 -c 2 /dev/null && echo "OK"
```

**出力例:**
```
**** RECORD Devices ****
card 1: Loopback [Loopback], device 0: Loopback PCM [Loopback PCM]
  Subdevices: 8/8
  Subdevice #0: subdevice #0
  Subdevice #1: subdevice #1
  ...
```

## 自動ロード設定（再起動永続化）

Jetson が起動するたび `snd-aloop` が自動でロードされるよう設定します。

### 方法1: `/etc/modprobe.d/` に設定ファイルを作成（推奨）

最も標準的な方法です。

#### Step 1: 設定ファイル作成

```bash
sudo tee /etc/modprobe.d/snd-aloop.conf > /dev/null << 'EOF'
# Enable ALSA Loopback for Magic Box audio pipeline
# Required for RTP → Loopback → GPU Convolution → DAC
options snd_aloop enable=1,1
alias snd-aloop snd_aloop
EOF
```

**設定の詳細:**
- `enable=1,1`: 2つのループバックデバイスを有効化（安全性向上のため複数確保）

#### Step 2: mkinitramfs で初期ramfsに含める（Jetson標準構成）

```bash
# 確認（含まれているか）
grep -r snd_aloop /etc/modprobe.d/ && echo "設定ファイルあり"

# 初期ramfs更新（オプション、通常は不要）
sudo update-initramfs -u
```

#### Step 3: 検証

```bash
# Jetson再起動
sudo reboot

# 再起動後、自動ロード確認
sleep 30  # 起動待機
lsmod | grep snd_aloop

# ALSA が認識しているか確認
arecord -l | grep Loopback
```

**期待される出力:**
```
snd_aloop              20480  0
```

### 方法2: systemd サービスで自動ロード（代替方法）

再起動時に systemd がモジュールをロードする方法です。

#### Step 1: systemd サービスファイル作成

```bash
sudo tee /etc/systemd/system/snd-aloop-load.service > /dev/null << 'EOF'
[Unit]
Description=Load snd-aloop kernel module for Magic Box
After=network-online.target
Wants=network-online.target
Before=docker.service

[Service]
Type=oneshot
ExecStart=/sbin/modprobe snd_aloop enable=1,1
ExecStop=/sbin/modprobe -r snd_aloop
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

#### Step 2: サービス有効化

```bash
# サービス有効化
sudo systemctl enable snd-aloop-load.service

# 即座にロード（テスト）
sudo systemctl start snd-aloop-load.service

# ステータス確認
sudo systemctl status snd-aloop-load.service

# ロード確認
lsmod | grep snd_aloop
```

**期待される出力:**
```
● snd-aloop-load.service - Load snd-aloop kernel module for Magic Box
   Loaded: loaded (/etc/systemd/system/snd-aloop-load.service; enabled; ...)
   Active: active (exited) since XXX
```

#### Step 3: Jetson再起動で自動ロード確認

```bash
sudo reboot

# 再起動後
sleep 30
lsmod | grep snd_aloop
arecord -l | grep Loopback
```

## docker-compose 統合

### Docker起動前の確認チェックリスト

docker-compose で コンテナを起動する前に、**必ず以下を確認**してください：

```bash
#!/bin/bash
# snd-aloop チェックスクリプト

echo "=== snd-aloop ロード状況 ==="
if lsmod | grep -q snd_aloop; then
    echo "✓ snd_aloop モジュルロード済み"
else
    echo "✗ snd_aloop モジュルが未ロード"
    echo "  → 以下のいずれかを実行:"
    echo "     sudo modprobe snd_aloop"
    echo "     または"
    echo "     sudo systemctl start snd-aloop-load.service"
    exit 1
fi

echo ""
echo "=== ALSA ループバックデバイス確認 ==="
if arecord -l | grep -q Loopback; then
    echo "✓ Loopback デバイス認識"
    arecord -l | grep -A 3 Loopback
else
    echo "✗ Loopback デバイスが認識されていない"
    exit 1
fi

echo ""
echo "=== /dev/snd マウント確認 ==="
ls -la /dev/snd/ | head -5
```

### docker-compose.yml の例

```yaml
version: '3.8'

services:
  gpu-upsampler:
    image: magic-box:latest
    container_name: magic-box
    runtime: nvidia

    # ⚠️ 重要: /dev/snd をコンテナに渡す
    devices:
      - /dev/snd:/dev/snd          # ALSA デバイス全体
      - /dev/hwrng:/dev/hwrng      # 乱数生成（オプション）

    # ALSA が /dev/snd を正しく認識するため、group IDを指定
    group_add:
      - audio                       # audio グループに追加

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ALSA_CONFIG_DIR=/etc/alsa   # コンテナ内ALSA設定

    volumes:
      - /etc/alsa:/etc/alsa:ro      # ホストのALSA設定を共有（読み取り専用）
      - /etc/asound.conf:/etc/asound.conf:ro
      - ./data:/data                # EQ係数など

    ports:
      - "8080:8080"                 # Web UI

    # ホストネットワークスタック使用（RTP受信用）
    network_mode: host

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/status"]
      interval: 10s
      timeout: 5s
      retries: 3

    restart: unless-stopped

networks:
  default:
    driver: bridge
```

**重要な設定:**

| 設定 | 理由 |
|------|------|
| `devices: /dev/snd:/dev/snd` | **必須**: ALSAデバイスをコンテナに渡す |
| `group_add: [audio]` | コンテナプロセスが audio グループで実行される |
| `volumes: /etc/alsa` | ホストのALSA設定をコンテナと共有（オプション） |
| `network_mode: host` | RTP受信でネットワークバッファを最小化 |

### docker-compose 起動フロー

```bash
# 1. snd-aloop ロード確認
lsmod | grep snd_aloop || sudo systemctl start snd-aloop-load.service

# 2. ALSA デバイス確認
arecord -l | grep Loopback

# 3. docker-compose 起動
docker-compose up -d

# 4. コンテナ内でALSAが認識しているか確認
docker-compose exec gpu-upsampler arecord -l

# 5. ログ確認
docker-compose logs -f gpu-upsampler
```

## トラブルシューティング

### Issue: `hw:Loopback` が見つからない

```bash
# 確認手順
echo "1. モジュール確認:"
lsmod | grep snd_aloop

echo "2. modprobe ロード試行:"
sudo modprobe snd_aloop

echo "3. 再度確認:"
lsmod | grep snd_aloop

echo "4. ALSA 再スキャン:"
sudo systemctl restart alsa-restore

echo "5. デバイス一覧:"
arecord -l
aplay -l
```

### Issue: コンテナ内で `/dev/snd` が見つからない

```bash
# ホストで確認
ls -la /dev/snd/

# docker-compose.yml で確認
grep -A 5 "devices:" docker-compose.yml

# 期待される出力: devices: - /dev/snd:/dev/snd

# 修正後、コンテナ再起動
docker-compose down
docker-compose up -d
```

### Issue: コンテナ内で `hw:Loopback` が見つからない

```bash
# コンテナ内で確認
docker-compose exec gpu-upsampler bash

# コンテナ内:
arecord -l
aplay -l

# group_add: [audio] が設定されているか確認
docker-compose exec gpu-upsampler id

# 出力例: groups=...,29(audio),...
```

### Issue: XRUN や アンダーラン が発生

```bash
# ALSA ロギングを有効化（ホスト側）
ALSA_CONFIG_DEBUG=1 aplay /dev/zero

# カーネルログ確認
sudo dmesg | tail -20

# オプション: /etc/modprobe.d/snd-aloop.conf で追加設定
sudo tee /etc/modprobe.d/snd-aloop.conf > /dev/null << 'EOF'
options snd_aloop enable=1,1 pcm_substreams=8
options snd_aloop max_devices=2
EOF

# 再ロード
sudo modprobe -r snd_aloop
sudo modprobe snd_aloop enable=1,1 pcm_substreams=8
```

## README/docker-compose 注記テンプレート

以下を **README.md** と **docker-compose.yml** に追加してください：

### README.md に追加

```markdown
## 環境別セットアップ

### Jetson Orin Nano（本番環境）

#### 前提: snd-aloop のロード

Magic Box のオーディオパイプラインは ALSA Loopback (`snd-aloop`) を必要とします。

**Jetson ホスト側で以下を実行してください:**

```bash
# 方法1: 手動ロード（テスト用）
sudo modprobe snd_aloop

# 方法2: 永続的にロード（推奨）
sudo tee /etc/modprobe.d/snd-aloop.conf > /dev/null << 'EOF'
options snd_aloop enable=1,1
EOF

# 再起動後も保持される確認
lsmod | grep snd_aloop
```

**詳細は [Jetson snd-aloop セットアップガイド](docs/setup/jetson_snd_aloop.md) を参照してください。**

#### docker-compose 起動

```bash
# 1. snd-aloop ロード確認
lsmod | grep snd_aloop || (echo "snd-aloop未ロード" && exit 1)

# 2. 起動
docker-compose up -d

# 3. ステータス確認
docker-compose ps
docker-compose logs gpu-upsampler
```
```

### docker-compose.yml に注記を追加

```yaml
version: '3.8'

# ⚠️ 注意: Jetsonホストで事前に snd-aloop をロードしてください
# 詳細: docs/setup/jetson_snd_aloop.md
#
# sudo modprobe snd_aloop
# または
# sudo systemctl start snd-aloop-load.service

services:
  gpu-upsampler:
    # ... (既存の設定)

    # ⚠️ 重要: /dev/snd をマウント（ALSA Loopback 使用に必須）
    devices:
      - /dev/snd:/dev/snd
```

## まとめ

| 項目 | 内容 |
|------|------|
| **必須手順** | `sudo modprobe snd_aloop` または `/etc/modprobe.d/snd-aloop.conf` 作成 |
| **永続化** | systemd サービス または modprobe.d の設定ファイル |
| **確認コマンド** | `lsmod \| grep snd_aloop` + `arecord -l` |
| **docker-compose** | `devices: - /dev/snd:/dev/snd` + `group_add: [audio]` |
| **トラブル** | [トラブルシューティング](#トラブルシューティング) を参照 |

---

**参考資料:**
- [ALSA Loopback Documentation](https://www.kernel.org/doc/html/latest/sound/designs/)
- [Jetson Linux Documentation](https://docs.nvidia.com/jetson/jetson-linux/)
- [Docker デバイスマウント](https://docs.docker.com/engine/reference/commandline/run/#device)
