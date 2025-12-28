# Raspberry Pi 評価者向け導入ガイド（I2Sメイン / RTPフォールバック）

Issue: #1061（Epic: #1051）

Raspberry Pi 側の評価者導入を一本道化するためのガイドです。
**現在の主流運用は I2S（USB/UAC2入力 → I2S出力）**で、RTP は緊急フォールバックとして残しています。

関連:

- 運用手順（I2S/RTP切替・トラブルシュート）: `docs/jetson/i2s/ops_runbook.md`
- UAC2受け口の考え方/設定: `docs/setup/pi_bridge.md`

---

## 起動（評価者向け / ソース不要: runtime-only）

Pi では GHCR image を pull して起動します（**ソースコード不要**）。

```bash
#
# NOTE: 配布された Release Notes の指定がある場合は、image を環境変数で固定してください。
#   USB_I2S_BRIDGE_IMAGE=ghcr.io/...:<tag> RASPI_CONTROL_API_IMAGE=ghcr.io/...:<tag> \
#
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
```

停止:

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml down
```

---

## 起動（開発者向け / ローカルビルド）

```bash
# デフォルト: I2S (USB/UAC2 -> I2S) ブリッジを起動
docker compose -f raspberry_pi/docker-compose.yml up -d --build

# RTP を起動したい場合（レアケース）: profile を明示
docker compose -f raspberry_pi/docker-compose.yml --profile rtp up -d --build rtp-sender rtp-bridge jetson-proxy
```

---

## 接続（迷わないための固定値）

- **Jetson Web**: `http://192.168.55.1/`
- **Pi Control API**（Jetson → Pi）: `http://192.168.55.100:8081`（USB直結の典型値）
- **Pi → Jetson ステータス送信**: `http://192.168.55.1/i2s/peer-status`

> IP がズレる場合は Jetson 側は `MAGICBOX_PI_API_BASE`、Pi 側は `USB_I2S_STATUS_REPORT_URL` を上書きしてください。

## 必要パッケージ (Raspberry Pi OS / Debian 系)

```bash
sudo apt-get update
sudo apt-get install -y \
  gstreamer1.0-alsa gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  python3 python3-pip

# ZeroMQ ブリッジを使う場合
pip3 install --user pyzmq
```

> NOTE: 評価者は基本的に Docker で起動してください（推奨）。以降の CLI セクションはデバッグ/フォールバック向けです。

---

## RTP送出: 使い方 (Python CLI)

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

- 対応フォーマット: `S16_LE`, `S24_3LE`, `S32_LE`（payload は L16/L24/L32）
- 対応レート: 44.1k/48k 系の代表レート（44100/48000/88200/96000/176400/192000/352800/384000/705600/768000）
- `audioresample quality=10` でクロックドリフトを吸収し、RTCP で Jetson へフィードバックします。
- ALSA の実サンプルレートを自動検出し、caps/clock-rate に反映。検出結果は ZeroMQ ブリッジ用の JSON にも書き出します。レート変化を検知するとパイプラインを自動再起動して追従します。
- `--dry-run` または `RTP_SENDER_DRY_RUN=true` でパイプライン文字列だけを出力します。

### 主な環境変数

| 変数 | 既定値 | 説明 |
| ---- | ------ | ---- |
| `RTP_SENDER_DEVICE` | `hw:0,0` | ALSA キャプチャデバイス |
| `RTP_SENDER_HOST` | `192.168.55.1` | Jetson RTP 受信先ホスト |
| `RTP_SENDER_RTP_PORT` | `46000` | RTP 送信ポート |
| `RTP_SENDER_RTCP_PORT` | `46001` | Jetson へ送る RTCP ポート |
| `RTP_SENDER_RTCP_LISTEN_PORT` | `46002` | Jetson から受ける RTCP ポート |
| `RTP_SENDER_SAMPLE_RATE` | `44100` | サンプルレート (Hz, 自動検出のフォールバック) |
| `RTP_SENDER_AUTO_SAMPLE_RATE` | `true` | ALSA からレートを検出し caps/clock-rate に適用 |
| `RTP_SENDER_RATE_POLL_INTERVAL_SEC` | `2.0` | レート変化検知のポーリング間隔 (秒) |
| `RTP_SENDER_CHANNELS` | `2` | チャンネル数 |
| `RTP_SENDER_FORMAT` | `S24_3BE` | フォーマット (S16_BE/S24_3BE/S32_BE 推奨) |
| `RTP_SENDER_LATENCY_MS` | `100` | jitterbuffer latency (ms) |
| `RTP_SENDER_PAYLOAD_TYPE` | `96` | RTP Payload Type |
| `RTP_SENDER_DRY_RUN` | `false` | true でパイプライン出力のみ |
| `RTP_BRIDGE_STATS_PATH` | `/tmp/rtp_receiver_stats.json` | ZeroMQ STATUS 用に検出レート等を書き出すパス |

---

## Docker / Compose 補足（RTPフォールバック）

- `raspberry_pi/docker-compose.yml` は **デフォルトで I2S ブリッジ**を起動します。RTP 系は `profiles: ["rtp"]` のため、明示しない限り起動しません。
- RTP を使う場合は RTCP 付き RTP 送出 (`rtp-sender`) と ZeroMQ ブリッジ (`rtp-bridge`) を分離しています。`rtp-bridge` は `tcp://0.0.0.0:60000` で待ち受け、Jetson から到達できます（ポート 60000 を開ける）。
- `RTP_SENDER_*` を `.env` で上書きすると配信先・フォーマットを切り替えられます。デフォルトは BE (`S24_3BE`)。

## ZeroMQ ブリッジ (任意)

`python3 -m raspberry_pi.rtp_receiver` を Sidecar として起動すると、以下をファイル経由で連携できます。

- `RTP_BRIDGE_STATS_PATH` (既定 `/tmp/rtp_receiver_stats.json`): 送信側が書き出す統計を ZeroMQ から STATUS 参照
- `RTP_BRIDGE_LATENCY_PATH` (既定 `/tmp/rtp_receiver_latency_ms`): SET_LATENCY 受信時に書き戻し、送信側で適用

Magic Box Web UI からのレイテンシ変更を Pi に伝える場合に使用します。

> NOTE: デフォルトの待受エンドポイントは `tcp://0.0.0.0:60000` に変更しました。Jetson 側は `RTP_BRIDGE_ENDPOINT=tcp://raspberrypi.local:60000` などホスト名/IP を揃えてください。`docker-compose.yml` はポート 60000 を公開します。

> 再起動ポリシー: `docker-compose.yml` では `restart: always` を指定しています。裸運用する場合も systemd で `Restart=always` を付け、片側クラッシュ時も自動復帰させてください。

### I2S 制御プレーン (Issue #824)

I2S のレート/フォーマット/チャンネルを Pi-Jetson 間で同期させ、どちらかが切断中でも復帰後に共通パラメータになるまで capture を待機します。

- REP (Pi): `USB_I2S_CONTROL_ENDPOINT`（`config.env` で設定、既定: 空=無効）
- REQ (Jetson 側など): `USB_I2S_CONTROL_PEER`（`config.env` で設定、既定: 空=無効）
- 待機ポリシー: `USB_I2S_CONTROL_REQUIRE_PEER=true`（`config.env` で設定、既定: false）
- タイムアウト/ポーリング: `USB_I2S_CONTROL_TIMEOUT_MS` / `USB_I2S_CONTROL_POLL_INTERVAL_SEC`（`config.env`）

Jetson 側も `raspberry_pi/usb_i2s_bridge/control_agent.py` を `python3 -m raspberry_pi.usb_i2s_bridge.control_agent` で起動すると、同じ仕組みでステータスを提供できます。

### Pi 制御 API (Issue #940)

Pi 側に軽量の FastAPI を常駐させ、Jetson から USB 経由で制御します。

- Docker Compose では `raspi-control-api` サービスとして起動します。
- デフォルト bind は `RPI_CONTROL_BIND_INTERFACE`（既定 `usb0`）を自動検出し、失敗した場合は **`RPI_CONTROL_BIND_SUBNET`（既定 `192.168.55.0/24`）に属するIPを持つインターフェースを探索**します。
  - それでも検出に失敗した場合は到達不能になるため、プロセスを終了（非0）して restart で再試行します。
  - 環境によりインターフェース名や起動順が異なる場合は `RPI_CONTROL_BIND_HOST`（例: `192.168.55.100`）を明示してください。
- **ポート 80 は使用しない**（Jetson 側 nginx へ戻るため）。既定は `8081`。
- `raspi-control-api` は **docker.sock をマウントすることが前提**（再起動/反映に必須）。
- `usb0` へバインドさせるため **host network が前提**。
  - Compose 利用時は `.env` に `RPI_CONTROL_BIND_HOST=192.168.55.100` のように置くと確実です。

主なエンドポイント:

- `GET /raspi/api/v1/status` : bridge 状態 / rate / format / ch / xruns / last_error / uptime
- `GET /raspi/api/v1/config` : 現在の設定
- `PUT /raspi/api/v1/config` : 設定更新（更新後に再起動）
- `POST /raspi/api/v1/actions/restart` : bridge 再起動

設定反映について:

- **設定は `/var/lib/usb-i2s-bridge/config.env` のみ**を編集します（唯一の設定元）。
- `usb-i2s-bridge` は起動時にこのファイルを読み込みます。
- `raspi-control-api` が同ファイルを更新し、Docker 経由で `usb-i2s-bridge` コンテナを再起動します。
- 初回起動時は `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` を seed としてコピーします。

Jetson から叩く例:

```bash
curl http://192.168.55.100:8081/raspi/api/v1/status

curl -X PUT http://192.168.55.100:8081/raspi/api/v1/config \
  -H 'Content-Type: application/json' \
  -d '{"alsa_buffer_time_us": 100000, "alsa_latency_time_us": 10000}'
```

### Jetson Web(:80) へのステータス送信 (Issue #950)

別ポートを増やさずに Jetson 側へ状態（mode/rate/format/ch）を通知したい場合は、Pi 側で以下を設定します（任意）。

- `USB_I2S_STATUS_REPORT_URL`（例: `http://192.168.55.1/i2s/peer-status`）※`config.env` で設定
- `USB_I2S_STATUS_REPORT_TIMEOUT_MS`（既定 300、`config.env`）
- `USB_I2S_STATUS_REPORT_MIN_INTERVAL_SEC`（既定 1.0、`config.env`）

## 参考: 生の GStreamer コマンド

Python ラッパーの出力と同等の gst-launch 例です。

```bash
gst-launch-1.0 -e rtpbin name=rtpbin ntp-sync=true buffer-mode=synced latency=100 \
  alsasrc device=hw:0,0 ! audioresample quality=10 ! audioconvert ! \
  audio/x-raw,rate=44100,channels=2,format=S24BE ! rtpL24pay pt=96 ! \
  application/x-rtp,media=audio,clock-rate=44100,encoding-name=L24,payload=96,channels=2 ! \
  rtpbin.send_rtp_sink_0 \
  rtpbin.send_rtp_src_0 ! udpsink host=<jetson-ip> port=46000 sync=true async=false \
  rtpbin.send_rtcp_src_0 ! udpsink host=<jetson-ip> port=46001 sync=false async=false \
  udpsrc port=46002 ! rtpbin.recv_rtcp_sink_0
```

## 移行メモ

- `raspberry_pi/src`, `include`, `tests`, `CMakeLists.txt` を含む TCP/C++ 実装は削除しました。
- Jetson 側の受信は GStreamer RTP (`web/services/rtp_input.py`) に統一し、旧 `jetson_pcm_receiver` はアーカイブ扱いです。
