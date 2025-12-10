# Raspberry Pi PCM Bridge (prototype)

Raspberry Pi向けのPCMブリッジ雛形です。ALSAキャプチャとTCPクライアントのスタブを含み、リンクが通る最小構成を提供します。

## 必要パッケージ

Raspberry Pi OS / Debian系:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libasound2-dev
```

## ビルド手順

```bash
cmake -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Release
cmake --build raspberry_pi/build
```

Debugビルド例:

```bash
cmake -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Debug
cmake --build raspberry_pi/build
```

生成物:
- バイナリ: `raspberry_pi/build/rpi_pcm_bridge`

## 動作概要

- 起動時にPCMヘッダ（`PCMA` / version=1 / rate / ch / format）をTCP接続確立後に送信し、その後はALSAから読み取ったPCMを順次送出します。
- TCP切断時は無制限リトライし、再接続後にヘッダを再送してPCM送出を再開します。
- SIGINT/SIGTERM受信でALSAとソケットをクリーンにクローズして終了します。
- `--log-level` で `debug|info|warn|error` を選択（デフォルト: `warn`）。主要イベントを標準出力/標準エラーへ出力します。
- `--iterations` はテスト用。0以下で無限ループ、正の値を指定するとその回数で自動終了します。
- ALSA実測サンプルレート/チャネル数/フォーマットの変化やデバイス再列挙を検知すると、ALSAを再オープンし、TCP再接続とヘッダ再送で追従します。
- ALSAデバイスが一時的に消失しても、復活するまでバックオフしながら再オープンを繰り返し、復旧後にヘッダを再送して送出を再開します。
- `--device auto` を指定すると、起動時に入力可能な最初のALSAデバイスを自動選択します。安定運用する場合は `hw:CARD=<名前>,DEV=0` などカードIDを用いた安定名指定を推奨します。

## 実行例

ヘルプ・バージョン表示:

```bash
./raspberry_pi/build/rpi_pcm_bridge --help
./raspberry_pi/build/rpi_pcm_bridge --version
```

TCP送信先とキャプチャ設定を指定する例:

```bash
./raspberry_pi/build/rpi_pcm_bridge \
  --device hw:0,0 \
  --host 192.168.1.50 \
  --port 46001 \
  --rate 96000 \
  --format S24_3LE \
  --frames 4096
```

- 対応フォーマット: `S16_LE`, `S24_3LE`, `S32_LE`
- 対応レート: `44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000`
- ポート範囲外や未対応フォーマット/レート指定時は起動時にエラー終了します。
- XRUN発生時は `snd_pcm_prepare()` でリカバリし、ログへ出力します。
- `PCM_BRIDGE_*` 環境変数でデフォルト値を上書き可能です（Dockerで利用）。

## Docker 実行（Raspberry Pi上）

### 単体コンテナ（/dev/sndを直接渡す）

```bash
# プロジェクトルートでビルド
docker build -f raspberry_pi/Dockerfile -t rpi-pcm-bridge .

# ヘルプ表示だけ行う例
docker run --rm --device /dev/snd -e PCM_BRIDGE_MODE=help rpi-pcm-bridge

# 実際に送信待機を行う例
docker run --rm --device /dev/snd \
  --group-add audio \
  --restart unless-stopped \
  -e PCM_BRIDGE_HOST=192.168.55.1 \
  -e PCM_BRIDGE_PORT=46001 \
  -e PCM_BRIDGE_DEVICE=hw:0,0 \
  -e PCM_BRIDGE_RATE=48000 \
  -e PCM_BRIDGE_FORMAT=S16_LE \
  rpi-pcm-bridge
```

- 基本は `/dev/snd` と `audio` グループ付与で動作します。権限問題がある場合のみ `--privileged` を検討してください。
- `PCM_BRIDGE_MODE=version` でバージョン表示、`PCM_BRIDGE_MODE=help` でヘルプを表示します。

### docker compose + nginxリバースプロキシ

Jetsonの80番ポートにNginxでリバースプロキシを挟み、同一LAN内からWeb UIを見られるようにする例です。

```bash
# Raspberry Pi上で実行
docker compose -f raspberry_pi/docker-compose.yml up -d --build rtp-sender jetson-proxy

# ログ確認
docker compose -f raspberry_pi/docker-compose.yml logs -f rtp-sender
docker compose -f raspberry_pi/docker-compose.yml logs -f jetson-proxy

# 停止
docker compose -f raspberry_pi/docker-compose.yml down
```

- 環境変数 `JETSON_HOST` と `JETSON_PORT` でプロキシ先を変更できます（デフォルト: `jetson:80`）。
- 80番ポートで待ち受けるため、ホスト側で別のサービスが使用していないことを確認してください。必要なら `ports` を `8080:80` に変更して利用してください。
- `restart: always` で電源ロスト・Dockerデーモン再起動後も自動再立ち上げします（composeデフォルト設定）。

### 環境変数一覧（コンテナ）

- `PCM_BRIDGE_DEVICE` (既定: `hw:0,0`)
- `PCM_BRIDGE_HOST` (既定: `127.0.0.1`)
- `PCM_BRIDGE_PORT` (既定: `46001`)
- `PCM_BRIDGE_RATE` (既定: `48000`)
- `PCM_BRIDGE_FORMAT` (`S16_LE` | `S24_3LE` | `S32_LE`)
- `PCM_BRIDGE_FRAMES` (既定: `4096`)
- `PCM_BRIDGE_LOG_LEVEL` (`debug` | `info` | `warn` | `error`, 既定: `warn`)
- `PCM_BRIDGE_ITERATIONS` (`-1` で無限送信)
- `PCM_BRIDGE_MODE` (`run` | `help` | `version` | `rtp`) — `rtp` を指定すると GStreamer RTP 送出モードで起動
- `RTP_SENDER_DEVICE` (既定: `hw:0,0`)
- `RTP_SENDER_HOST` (既定: `jetson`)
- `RTP_SENDER_RTP_PORT` (既定: `46000`)
- `RTP_SENDER_RTCP_PORT` (既定: `46001`)
- `RTP_SENDER_RTCP_LISTEN_PORT` (既定: `46002`)
- `RTP_SENDER_PAYLOAD_TYPE` (既定: `96`)
- `RTP_SENDER_POLL_MS` (既定: `250`)
- `RTP_SENDER_LOG_LEVEL` (既定: `warn`)
- `RTP_SENDER_FORMAT` (`S16_LE` | `S24_3LE` | `S32_LE` を固定したい場合)
- `RTP_SENDER_NOTIFY_URL` (レート変更時に HTTP POST を送る先)
- `RTP_SENDER_DRY_RUN` (`true`/`1` でパイプライン起動せず文字列だけ確認)
- `RTP_BRIDGE_ENDPOINT` (ZeroMQ REP 待ち受け。既定: `ipc:///tmp/rtp_receiver.sock`)
- `RTP_BRIDGE_STATS_PATH` (STATUS 参照用に監視する JSON パス。既定: `/tmp/rtp_receiver_stats.json`)
- `RTP_BRIDGE_LATENCY_PATH` (SET_LATENCY 受信時に書き出すパス。既定: `/tmp/rtp_receiver_latency_ms`)
- `RTP_BRIDGE_TIMEOUT_MS` (ZeroMQ send/recv タイムアウト。既定: `5000`)
- `RTP_BRIDGE_POLL_INTERVAL_SEC` (統計ファイルのポーリング間隔。既定: `1.0`)

## 手動テスト（null sink/loopback + nc）

Jetson側TCPサーバが無くても、ローカルのALSA loopback + `nc` でヘッダとPCM転送を検証できます。

1. ALSAループバックを有効化
   `sudo modprobe snd-aloop pcm_substreams=2`
2. TCP受信を起動（別シェル）
   `nc -l -p 46001 > /tmp/pcm_dump.raw`
3. ブリッジを起動（Loopbackキャプチャを指定）
   ```bash
   ./raspberry_pi/build/rpi_pcm_bridge \
     --device hw:Loopback,1 \
     --host 127.0.0.1 \
     --port 46001 \
     --rate 48000 \
     --format S16_LE \
     --frames 1024 \
     --log-level debug
   ```
4. 任意の音声をLoopback再生側へ流す（別シェル）
   `speaker-test -D hw:Loopback,0 -c 2 -r 48000 -F S16_LE`
   または `aplay -D hw:Loopback,0 /path/to/test.wav`
5. 受信を確認
   - `hexdump -C /tmp/pcm_dump.raw | head` で先頭16バイトが `50 43 4d 41` (`PCMA`) になっていることを確認。
   - ファイルサイズが再生に合わせて増えていくことを確認。
   - ブリッジは `Ctrl+C` で終了（SIGINTでクリーンに停止）。

## GStreamer RTP 送出（RTCP付きでクロック同期）

デフォルトポート: `46000/udp` (RTP), `46001/udp` (RTCP to Jetson), `46002/udp` (RTCP from Jetson)。送受信とも RTCP を流し、Jetson 側が送信側クロックに同期します。サンプルレート初期値は 44.1kHz 固定、`audioresample quality=10` で微小ドリフトを吸収します。

送信（Raspberry Pi 側）例:

```bash
gst-launch-1.0 -e rtpbin name=rtpbin ntp-sync=true buffer-mode=sync \
  alsasrc device=hw:0,0 ! audioresample quality=10 ! audioconvert ! \
  audio/x-raw,rate=44100,channels=2,format=S24LE ! rtpL24pay pt=96 ! rtpbin.send_rtp_sink_0 \
  rtpbin.send_rtp_src_0 ! udpsink host=<jetson-ip> port=46000 sync=true async=false \
  rtpbin.send_rtcp_src_0 ! udpsink host=<jetson-ip> port=46001 sync=false async=false \
  udpsrc port=46002 ! rtpbin.recv_rtcp_sink_0
```

- 16bit/32bit を送りたい場合は `rtpL16pay` / `rtpL32pay` と `format=S16LE/S32LE` に差し替えてください（pt は 96 のままで共有して問題ありません）。
- レイテンシ調整は Jetson 側（Magic Box コンテナ内 `/api/rtp-input/config` の `latency_ms`、デフォルト100ms）で行います。

## rpi_rtp_sender（RTP送出・自動レート追従）

- ALSAデバイスの `hw_params` (`/proc/asound/cardX/pcmYc/sub0/hw_params`) をポーリングし、現在のサンプルレート/フォーマット/チャネル数を検出します。
- レート・フォーマットの変化を検知すると、GStreamer RTP送出パイプラインを EOS→再生成して `clock-rate` を更新します。
- 対応フォーマット: `S16_LE` / `S24_3LE` / `S32_LE`（24bitは `rtpL24pay`）
- 対応レート: 44.1k/48k系の全10レート（44.1/48/88.2/96/176.4/192/352.8/384/705.6/768 kHz）
- 既定ポート: RTP 46000/UDP, RTCP to Jetson 46001/UDP, RTCP from Jetson 46002/UDP
- `--rate-notify-url` を指定すると、レート変更検知時に `curl -X POST rate=<Hz>&channels=<n>&format=<enum>` を送信します（Jetson側監視との簡易連携用）。

### 使い方

```bash
# ビルド（例）
cmake -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Release
cmake --build raspberry_pi/build --target rpi_rtp_sender

# Dry-run でパイプライン文字列のみ確認
./raspberry_pi/build/src/rpi_rtp_sender --device hw:0,0 --host 192.168.55.1 --dry-run

# 実行（自動レート追従）
./raspberry_pi/build/src/rpi_rtp_sender \
  --device hw:0,0 \
  --host 192.168.55.1 \
  --rtp-port 46000 \
  --rtcp-port 46001 \
  --rtcp-listen-port 46002 \
  --poll-ms 200 \
  --log-level info
```

主なCLI/環境変数:

- `--device` / `RTP_SENDER_DEVICE` : ALSAデバイス（例 `hw:0,0`）
- `--host` / `RTP_SENDER_HOST` : Jetson RTP受信側ホスト名/IP
- `--rtp-port` / `RTP_SENDER_RTP_PORT` : RTP送信ポート
- `--rtcp-port` / `RTP_SENDER_RTCP_PORT` : Jetsonへ送るRTCPポート
- `--rtcp-listen-port` / `RTP_SENDER_RTCP_LISTEN_PORT` : Jetsonから受けるRTCPポート
- `--payload-type` / `RTP_SENDER_PAYLOAD_TYPE` : RTP PT（既定96）
- `--format` / `RTP_SENDER_FORMAT` : フォーマットを固定したい場合に指定
- `--poll-ms` / `RTP_SENDER_POLL_MS` : hw_paramsポーリング間隔（既定250ms）
- `--rate-notify-url` / `RTP_SENDER_NOTIFY_URL` : レート変更通知先URL（任意）
- `--log-level` / `RTP_SENDER_LOG_LEVEL`
- `--dry-run` / `RTP_SENDER_DRY_RUN`
