# 評価者向け QuickStart（オンボーディング）

Issue: #1063（Epic: #1051）

このドキュメントは、評価者が **迷わず試用し、問題報告まで到達**するための最短手順です（ソースコード不要）。

---

## まず読む（入口）

- Jetson: [Jetson 評価者向け導入ガイド（ソース不要 / Docker）](./evaluator_guide_docker.md)
- Docker: [Docker（評価者向け / ソース不要）](../../docker/README.evaluator.md)
- Raspberry Pi（RTP sender / usb-i2s-bridge / 既知注意点）: [Raspberry Pi README](../../raspberry_pi/README.md)

---

## 構成A: Jetsonのみ（最短）

Jetson で評価版を起動して、Web UI を開きます。

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
docker compose -f jetson/docker-compose.jetson.runtime.yml logs -f
```

- Web UI: `http://192.168.55.1/`（デフォルト。LAN 公開は `MAGICBOX_PUBLISH_IP=0.0.0.0` を明示して上書き）

---

## 構成B: Raspberry Pi を併用（入力ブリッジ / RTP）

評価環境によって Pi の役割が変わるため、該当する方を選びます。

### B-1) Pi: USB/UAC2入力 → I2S出力（usb-i2s-bridge）

Pi 側（runtime-only）:

```bash
cd /path/to/magicbox-root
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
```

初期設定（必須）:

- `usb-i2s-bridge` の設定は `USB_I2S_CONFIG_PATH`（既定: `/var/lib/usb-i2s-bridge/config.env`）に集約されます
- 少なくとも次は環境に合わせて調整してください:
  - `USB_I2S_CAPTURE_DEVICE`（UAC2 “受け口” の ALSA device。例: `hw:CARD=UAC2Gadget,DEV=0`）
  - `USB_I2S_PLAYBACK_DEVICE`（I2S 出力。典型: `hw:0,0`）
  - `USB_I2S_PASSTHROUGH=true`（推奨）
  - `USB_I2S_PREFERRED_FORMAT=S32_LE`（推奨）

確認コマンド例:

```bash
arecord -l
aplay -l
cat /proc/asound/cards
```

> 詳細は [Raspberry Pi README](../../raspberry_pi/README.md) と `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` を参照してください。

### B-2) Pi: ALSAキャプチャ → RTP/RTCP（rtp_sender）

この構成は、Pi が ALSA からキャプチャした音声を **RTP(L16/L24/L32)+RTCP** で Jetson に送る場合です。

重要（必須）:

- **RTP L16/L24/L32 はネットワークバイトオーダー（BE）前提**
  - 送信・受信で `S24BE` 等に統一してください（例: `S24_3BE`）
- **payload type は 96**
- 送信側（Pi）は変換/リサンプルの遅延吸収のため、`alsasrc` の `buffer-time/latency-time` を明示し、**`queue` を必ず挟む**（`raspberry_pi/rtp_sender.py` はこの方針で構成）

Pi 側（Python CLI 例）:

```bash
python3 -m raspberry_pi.rtp_sender \
  --device hw:0,0 \
  --host 192.168.55.1 \
  --sample-rate 44100 \
  --channels 2 \
  --format S24_3BE \
  --rtp-port 46000 \
  --rtcp-port 46001 \
  --rtcp-listen-port 46002 \
  --latency-ms 100
```

Jetson 側（RTP受信を有効化）:

`docker/jetson/docker-compose.jetson.runtime.yml` は UDP/46000-46002 を公開していますが、RTP受信はデフォルト無効です。
以下のように **明示的に有効化**して起動してください。

```bash
cd docker
MAGICBOX_ENABLE_RTP=true MAGICBOX_RTP_AUTOSTART=true \
  docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
```

注意（RTPが届かないとき）:

- `docker/jetson/docker-compose.jetson.runtime.yml` は **RTP/RTCP の待受 bind 先も** `MAGICBOX_PUBLISH_IP` に従います
  - デフォルト: `192.168.55.1`（USB gadget）
  - Pi が USB gadget のサブネット外（LAN側）から送る場合は、明示的に変更してください:

```bash
cd docker
MAGICBOX_PUBLISH_IP=0.0.0.0 MAGICBOX_ENABLE_RTP=true MAGICBOX_RTP_AUTOSTART=true \
  docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
```

---

## 動作確認（最低限）

- [ ] Web UI が開ける（`http://192.168.55.1/`）
- [ ] 音が出る（無音/片ch/歪み/クリックがない）
- [ ] 入力レートが想定通りに見える（44.1k 系 / 48k 系）
- [ ] EQ の適用/解除ができる

次に、評価者向けチェックリストで一通り確認してください:

- [評価者向けテストチェックリスト](./quality/evaluator_test_checklist.md)

---

## ログ取得（不具合報告の前に）

### Jetson（コンテナログ + 環境情報）

手順は `./evaluator_guide_docker.md` の「ログ取得」をそのまま使ってください。

### Raspberry Pi

- runtime compose の場合:

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs --since 1h --no-color > raspi.log
```

- `rtp_sender` の場合:
  - 実行ログ（標準出力）をテキストで保存してください
  - `arecord -l` / `aplay -l` / `cat /proc/asound/cards` の結果も添付してください

---

## フィードバックの書き方（テンプレ）

Issue に貼れるテンプレを用意しています（コピペ用）:

- [フィードバックテンプレ（評価者向け）](./evaluator_feedback_template.md)
