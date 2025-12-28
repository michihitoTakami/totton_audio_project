# 評価者向けフィードバックテンプレ（コピペ用）

Issue: #1063（Epic: #1051）

このテンプレを GitHub Issue に貼り付けてください。

---

## 概要（1-2行）

- 何が起きたか:
- どの構成か（USB / RTP）:

---

## 環境

### Jetson / Magic Box 側

- 機種:
- JetPack / L4T:（`cat /etc/nv_tegra_release`）
- 取得した Docker 情報:
  - `docker version`:
  - `docker compose version`:
- DAC:
- ヘッドホン:
- 接続:
  - 入力:（USB gadget / RTP）
  - 出力:（DAC 型番）
- 起動コマンド/compose:
  - `cd docker && docker compose -f jetson/docker-compose.jetson.runtime.yml up -d`
  - RTP 利用時の上書き（あれば）: `MAGICBOX_ENABLE_RTP=... MAGICBOX_RTP_AUTOSTART=...`

### Raspberry Pi 側（使用している場合）

- 機種 / OS:
- 役割:（usb-i2s-bridge / rtp_sender / 併用）
- ALSA device:
  - capture（例: `hw:0,0`）:
  - playback（I2S, 例: `hw:0,0`）:
- 送信（RTPの場合）:
  - format（例: `S24_3BE`）:
  - payload type（例: `96`）:
  - ports（RTP/RTCP/RTCP listen）:

---

## 再現手順

1.
2.
3.

---

## 期待する挙動

-

---

## 実際の挙動

-

---

## 発生頻度

- 例: 毎回 / たまに（1時間に1回） / 初回のみ

---

## ログ/添付（可能な範囲で）

### Jetson

- `magicbox.log`（例: `docker compose ... logs --since 1h --no-color > magicbox.log`）
- `env.txt`（例: `uname -a` / `cat /etc/nv_tegra_release` / `docker info` / `docker compose ps`）

### Raspberry Pi（使用時）

- `raspi.log`（composeログ）
- `rtp_sender` の標準出力ログ（RTP使用時）
- `arecord -l` / `aplay -l` / `cat /proc/asound/cards`

---

## 補足（任意）

- 直前にやった操作（レート切替/EQ適用/ケーブル抜き差し等）:
- 既知トラブルへの対処を試したか（何を試したか）:
