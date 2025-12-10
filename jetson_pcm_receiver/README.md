# Jetson RTP Receiver (GStreamer)

`jetson_pcm_receiver` の TCP/C++ 実装は廃止され、Jetson 側は GStreamer ベースの RTP 受信に一本化しました。受信パイプラインは Magic Box Web サーバの `RtpReceiverManager`（`web/services/rtp_input.py`）が管理し、HTTP API から開始/停止・設定変更できます。

## 受信の起動

```bash
# Web API 経由
curl -X POST http://localhost:8000/api/rtp-input/start

# 設定更新の例（44.1kHz/L24/RTCP付き）
curl -X PUT http://localhost:8000/api/rtp-input/config \
  -H 'Content-Type: application/json' \
  -d '{
    "port": 46000,
    "rtcp_port": 46001,
    "rtcp_send_port": 46002,
    "sample_rate": 44100,
    "channels": 2,
    "encoding": "L24",
    "latency_ms": 100
  }'
```

エンドポイント一覧:
- `GET  /api/rtp-input/status` – 稼働状態と現在の設定
- `POST /api/rtp-input/start` – パイプライン起動
- `POST /api/rtp-input/stop` – 停止
- `PUT  /api/rtp-input/config` – 設定更新（再起動は任意）

## 既定設定 (環境変数)

`web/services/rtp_input.py` は以下の環境変数を読み込みます。未指定時はデフォルト値で起動します。

| 変数 | 既定値 | 説明 |
| ---- | ------ | ---- |
| `MAGICBOX_RTP_PORT` | `46000` | RTP 受信ポート |
| `MAGICBOX_RTP_RTCP_PORT` | `46001` | RTCP 受信ポート |
| `MAGICBOX_RTP_RTCP_SEND_PORT` | `46002` | 送信側へ返す RTCP ポート |
| `MAGICBOX_RTP_SAMPLE_RATE` | `44100` | サンプルレート |
| `MAGICBOX_RTP_CHANNELS` | `2` | チャンネル数 |
| `MAGICBOX_RTP_ENCODING` | `L24` | `L16` / `L24` / `L32` |
| `MAGICBOX_RTP_LATENCY_MS` | `100` | jitterbuffer latency |
| `MAGICBOX_RTP_DEVICE` | `hw:Loopback,0,0` | ALSA 出力デバイス |
| `MAGICBOX_RTP_QUALITY` | `10` | `audioresample` quality (0-10) |

## パイプライン概要

- `udpsrc` → `rtpbin` → `rtpL16/24/32depay` → `audioconvert` → `audioresample quality=10` → `alsasink`
- RTCP を port+1 (受信) / port+2 (送信) でやり取りし、送信側クロックに同期します。
- 送信側 (Raspberry Pi) には `raspberry_pi/rtp_sender.py` を利用してください。

## 廃止されたもの

- `jetson_pcm_receiver/src`, `include`, `tests`, `CMakeLists.txt` 配下の TCP 実装は削除済みです。
- `docker/jetson_pcm_receiver/` のビルドフローも非推奨となりました。Jetson 本体のコンテナ内で Web サービスを起動する構成に移行してください。
