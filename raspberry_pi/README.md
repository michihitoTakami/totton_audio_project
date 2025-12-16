# Raspberry Pi RTP Sender (GStreamer)

Raspberry Pi 上で ALSA キャプチャを GStreamer RTP (+RTCP) で Jetson へ送出するための最小構成です。TCP ベースの C++ 実装は廃止し、RTCP 同期付きの GStreamer パイプラインを正式ルートとしました。

- 送信側: `raspberry_pi/rtp_sender.py`（gst-launch ラッパー、**デフォルトは BE (S24_3BE)**）
- 連携: `raspberry_pi/rtp_receiver`（ZeroMQ ブリッジで統計/レイテンシ共有、任意）
- 受信側: Jetson の `rtp_input` サービス（`web/services/rtp_input.py`）

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

## 使い方 (Python CLI)

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

## Docker / Compose (Raspberry Pi 上)

```bash
# デフォルト: I2S (USB/UAC2 -> I2S) ブリッジを起動
docker compose -f raspberry_pi/docker-compose.yml up -d --build

# RTP を起動したい場合（レアケース）: profile を明示
docker compose -f raspberry_pi/docker-compose.yml --profile rtp up -d --build rtp-sender rtp-bridge jetson-proxy
```

- `raspberry_pi/docker-compose.yml` は **デフォルトで I2S ブリッジ**を起動します。RTP 系は `profiles: ["rtp"]` のため、明示しない限り起動しません。
- RTP を使う場合は RTCP 付き RTP 送出 (`rtp-sender`) と ZeroMQ ブリッジ (`rtp-bridge`) を分離しています。`rtp-bridge` は `tcp://0.0.0.0:60000` で待ち受け、Jetson から到達できます（ポート 60000 を開ける）。
- `RTP_SENDER_*` を `.env` で上書きすると配信先・フォーマットを切り替えられます。デフォルトは BE (`S24_3BE`)。サンプルレートは自動検出が有効で、`RTP_BRIDGE_STATS_PATH` に検出値を書き出します。

## ZeroMQ ブリッジ (任意)

`python3 -m raspberry_pi.rtp_receiver` を Sidecar として起動すると、以下をファイル経由で連携できます。

- `RTP_BRIDGE_STATS_PATH` (既定 `/tmp/rtp_receiver_stats.json`): 送信側が書き出す統計を ZeroMQ から STATUS 参照
- `RTP_BRIDGE_LATENCY_PATH` (既定 `/tmp/rtp_receiver_latency_ms`): SET_LATENCY 受信時に書き戻し、送信側で適用

Magic Box Web UI からのレイテンシ変更を Pi に伝える場合に使用します。

> NOTE: デフォルトの待受エンドポイントは `tcp://0.0.0.0:60000` に変更しました。Jetson 側は `RTP_BRIDGE_ENDPOINT=tcp://raspberrypi.local:60000` などホスト名/IP を揃えてください。`docker-compose.yml` はポート 60000 を公開します。

> 再起動ポリシー: `docker-compose.yml` では `restart: always` を指定しています。裸運用する場合も systemd で `Restart=always` を付け、片側クラッシュ時も自動復帰させてください。

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
