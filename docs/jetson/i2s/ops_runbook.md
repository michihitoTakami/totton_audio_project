# I2S移行 運用手順（Runbook, Issue #827）

## 目的
I2S移行を運用に載せるため、以下を「手順書だけで第三者が再現できる」レベルで整理する。

- Pi/Jetsonそれぞれの起動/停止（systemd/Docker方針含む）
- RTP→I2S切替手順
- I2S→RTPフォールバック手順（緊急時の逃げ道）
- 典型トラブル（無音/片ch/歪み/クリック）時のチェックリスト

## 前提（推奨構成）
- **Jetson**: `docker/jetson/docker-compose.jetson.yml` の `magicbox` コンテナで運用（本番想定）
- **Raspberry Pi**: `raspberry_pi/docker-compose.yml` で運用し、常駐は systemd ユニットでラップ
  - ユニット導入: `scripts/deployment/setup-pi-usb-i2s-bridge.sh`

> NOTE: Jetsonの「開発用autostart」は `docs/jetson/dev-autostart.md`（開発ワークツリー向け）を参照。

---

## 用語
- **I2S運用**: Piが USB(PC) 入力を取り込み I2S(TX) で Jetson へ送る（`usb-i2s-bridge`）
- **RTP運用**: Piが RTP で Jetson へ送る（`rtp-sender`）、Jetsonは RTP を受信して Loopback へ出す（`rtp_input`）

---

## 設定ファイルの場所（重要）

### Jetson（magicboxコンテナ）
- 設定実体: `/opt/magicbox/config/config.json`（Docker volume `magicbox-config`）
- 参照用シンボリックリンク: `/opt/magicbox/config.json`（上記への symlink）
- RTP APIの露出/自動起動:
  - `MAGICBOX_ENABLE_RTP=true` で RTP関連APIが有効化
  - `MAGICBOX_RTP_AUTOSTART=true` でWeb起動時にRTP受信を自動開始

### Raspberry Pi（usb-i2s-bridge）
- 設定実体: `/var/lib/usb-i2s-bridge/config.env`
  - 初回はコンテナ起動時に `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` を seed して作られる
- 状態ファイル: `/var/run/usb-i2s-bridge/status.json`

---

## 起動手順（I2S運用）

### Jetson（Docker Compose）
事前にJetson上でビルド済みであること（詳細は `docs/setup/build.md` / `docker/README.md`）。

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml up -d --build
docker compose -f jetson/docker-compose.jetson.yml logs -f
```

確認（例）:
- Web: `http://192.168.55.1/` が開ける
- コンテナ内設定: `docker compose -f jetson/docker-compose.jetson.yml exec magicbox ls -l /opt/magicbox/config.json`
- I2S入力が有効: `docker compose -f jetson/docker-compose.jetson.yml exec magicbox jq '.i2s.enabled, .i2s.device' /opt/magicbox/config/config.json`

### Raspberry Pi（Docker Compose + systemd常駐）
まずは手動で起動確認（その後systemd導入推奨）。

```bash
docker compose -f raspberry_pi/docker-compose.yml up -d --build
docker compose -f raspberry_pi/docker-compose.yml logs -f
```

常駐化（推奨）:

```bash
sudo ./scripts/deployment/setup-pi-usb-i2s-bridge.sh
sudo systemctl status usb-i2s-bridge.service
```

Pi側の動作確認:
- `/var/run/usb-i2s-bridge/status.json` の `mode` が `capture`（入力あり）または `silence`（入力なしだがI2S維持）になっている

---

## 停止手順（I2S運用）

### Jetson
```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml down
```

### Raspberry Pi
systemd導入済みなら:
```bash
sudo systemctl stop usb-i2s-bridge.service
```

手動運用なら:
```bash
docker compose -f raspberry_pi/docker-compose.yml down
```

---

## レート切替（44.1k系 ⇄ 48k系）

### 期待挙動
- **Pi**: `usb-i2s-bridge` が ALSA の `hw_params` を監視し、レート/フォーマット変化を検知して安全側（必要なら変換フォールバック）で再起動する
- **Jetson**: I2S capture が実レートを検知し、エンジン側に追従をスケジュールする（ログに `"[I2S] Detected input rate ... Scheduling rate follow."` が出る）

### 失敗したときの最短復旧
- Pi: `sudo systemctl restart usb-i2s-bridge.service`（または compose 再起動）
- Jetson: `docker compose -f jetson/docker-compose.jetson.yml restart magicbox`

---

## 切替手順: RTP → I2S

### 1) Jetson: I2S入力へ切替
`config.json` を I2S優先にし、必要ならコンテナ再起動する。

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml exec magicbox sh -lc '
  set -e
  jq ".i2s.enabled=true
      | .i2s.device=\"hw:APE,0\"
      | .i2s.sampleRate=0
      | .i2s.channels=2
      | .i2s.format=\"S32_LE\"
      | .i2s.periodFrames=1024
      | .loopback.enabled=false" /opt/magicbox/config/config.json > /tmp/config.json
  mv /tmp/config.json /opt/magicbox/config/config.json
'
docker compose -f jetson/docker-compose.jetson.yml restart magicbox
```

RTPを無効化する（不要なら）:
```bash
cd docker
MAGICBOX_ENABLE_RTP=false MAGICBOX_RTP_AUTOSTART=false \
  docker compose -f jetson/docker-compose.jetson.yml up -d
```

### 2) Pi: I2Sブリッジを起動（RTP系は止める）
RTP profile で起動していた場合は止め、通常起動へ戻す。

```bash
docker compose -f raspberry_pi/docker-compose.yml down
docker compose -f raspberry_pi/docker-compose.yml up -d --build
```

systemd運用なら:
```bash
sudo systemctl restart usb-i2s-bridge.service
```

---

## フォールバック手順: I2S → RTP（緊急）

### 0) 前提確認（Jetson側）
RTPは `MAGICBOX_ENABLE_RTP=true` のときのみAPIが有効になる。加えて、
`docker/jetson/docker-compose.jetson.yml` は UDP(46000/46001/46002) を公開済み（Issue #827）。

### 1) Jetson: loopback入力へ切替 + RTPを有効化

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml exec magicbox sh -lc '
  set -e
  jq ".i2s.enabled=false
      | .loopback.enabled=true
      | .loopback.device=\"hw:Loopback,1,0\"
      | .loopback.sampleRate=44100
      | .loopback.channels=2
      | .loopback.format=\"S32_LE\"
      | .loopback.periodFrames=1024" /opt/magicbox/config/config.json > /tmp/config.json
  mv /tmp/config.json /opt/magicbox/config/config.json
'

# RTP API有効化 + 自動起動（どちらも必要）
MAGICBOX_ENABLE_RTP=true MAGICBOX_RTP_AUTOSTART=true \
  docker compose -f jetson/docker-compose.jetson.yml up -d

# 念のため再起動（設定読み直し + autostart）
docker compose -f jetson/docker-compose.jetson.yml restart magicbox
```

自動起動しない運用なら（`MAGICBOX_RTP_AUTOSTART=false` の場合）:
```bash
curl -X POST http://192.168.55.1/api/rtp-input/start
```

### 2) Pi: I2Sブリッジを止めてRTP送出を開始
I2Sブリッジが動いたままだと ALSA capture を掴んで競合し得るため、先に停止する。

systemd運用なら:
```bash
sudo systemctl stop usb-i2s-bridge.service
```

RTP起動（profile指定）:
```bash
docker compose -f raspberry_pi/docker-compose.yml down
docker compose -f raspberry_pi/docker-compose.yml --profile rtp up -d --build rtp-sender rtp-bridge jetson-proxy
```

---

## 典型トラブル チェックリスト

### 無音
- Jetson:
  - `docker compose -f jetson/docker-compose.jetson.yml logs -f` に I2S/Loopback のエラーが出ていないか
  - I2S運用なら `"[I2S] Cannot open capture device"` が出ていないか（`i2s.device` / 配線 / APE設定）
  - RTP運用なら `GET /api/rtp-input/status` で `running=true` か
- Pi:
  - `/var/run/usb-i2s-bridge/status.json` の `mode` / `last_error` / `xruns`
  - `USB_I2S_CAPTURE_DEVICE` が実機の ALSA デバイスに合っているか（`/var/lib/usb-i2s-bridge/config.env`）

### 片ch
- `channels=2` になっているか（Jetson: `config.json`、Pi: `config.env`）
- I2S配線（LRCLK/DATA/GND）と左右設定（Left/Right）が合っているか

### 歪み（音割れ/ガリ）
- format不一致の疑い:
  - Jetson I2S: `i2s.format=S32_LE` を推奨（24in32運用）
  - RTP: `rtp_input` は ALSA 直前を `S32LE` に揃える（loopback側の format も合わせる）
- Pi側で passthrough が不安定な場合:
  - `USB_I2S_PASSTHROUGH=false`（preferred_formatへ変換して安定性優先）

### クリック/プチノイズ（XRUN）
- Pi側:
  - `USB_I2S_ALSA_BUFFER_TIME_US` / `USB_I2S_ALSA_LATENCY_TIME_US` を増やす（まず 2倍）
- Jetson側:
  - `i2s.periodFrames` を増やす（例: 1024→2048）
  - コンテナ環境変数 `MAGICBOX_WAIT_AUDIO_SECS` を増やし、起動時のデバイス未準備を避ける

---

## 関連
- `docs/jetson/i2s/i2s_migration_spec_820.md`（仕様）
- `raspberry_pi/README.md`（Pi側 compose / RTP sender / control api）
- `docs/jetson/troubleshooting.md`（Jetson全般のトラブルシュート）
