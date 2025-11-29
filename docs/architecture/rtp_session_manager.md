# RTP Session Manager (Issue #358)

## 概要

Jetson Orin Nano では PipeWire 依存を排除し、Data Plane (C++/CUDA) が直接 AES67 相当の
PCM/RTP ストリームを受信する必要がある。本ドキュメントでは `RtpSessionManager`
コンポーネントの設計と ZeroMQ API を定義する。

## 目的

1. ZeroMQ 経由で RTP セッションを開始 / 停止し、ALSADaemon へ PCM を供給する。
2. Unicast / Multicast / QoS (TTL, DSCP) を含むソケット初期化と RTCP 受信を行う。
3. PTP の同期状態を Data Plane に伝搬し、GPU 処理のブロック遅延を安定化する。
4. 監視しやすいテレメトリ (パケット数、ドロップ、SSRC 変化) を公開する。

## アーキテクチャ

```
┌────────────┐        ZeroMQ        ┌──────────────────────┐
│ Control UI │  ───────cmd───────>  │ ALSA Daemon (C++)    │
└────────────┘                     │  • RtpSessionManager  │
                                   │  • ConvolutionEngine  │
                                   └────────┬──────────────┘
                                            │ PCM (float stereo)
                                   ┌────────▼────────┐
                                   │ SoftMute / HRTF │
                                   └────────┬────────┘
                                            │
                                      ALSA Output
```

`RtpSessionManager` はマルチスレッドで動作し、各セッションごとに

- RTP 受信用スレッド (非ブロッキング `poll` + `recvfrom`)
- 任意の RTCP 受信用スレッド

を持つ。受信した RTP ペイロードは PCM32/PCM24/PCM16 → `float` に変換・デインタリーブされ、
`process_interleaved_block()` を通じて GPU パイプラインに供給される。

### セッション設定

`Network::SessionConfig` は以下のフィールドを持つ:

- `session_id`: 管理用 ID (必須)
- `bind_address`, `port`: 受信バインド
- `source_host`: 指定した送信元 IP のみ受け入れるフィルタ (任意)
- `multicast`, `multicast_group`, `interfaceName`, `ttl`, `dscp`
- `sample_rate`, `channels`, `bits_per_sample`, `payload_type`
- `target_latency_ms`, `watchdog_timeout_ms`, `telemetry_interval_ms`
- `enable_rtcp`, `rtcp_port`
- `enable_ptp`, `ptp_interface`, `ptp_domain`

## ZeroMQ API

| コマンド | Params | 説明 |
|----------|--------|------|
| `RTP_START_SESSION` | SessionConfig JSON | RTPセッションを作成し受信を開始する |
| `RTP_STOP_SESSION` | `{ "session_id": "foo" }` | 指定セッションを停止・破棄 |
| `RTP_LIST_SESSIONS` | なし | 現在のセッション一覧とメトリクス |
| `RTP_GET_SESSION` | `{ "session_id": "foo" }` | 単一セッションのメトリクス |
| `RTP_DISCOVER_STREAMS` | なし | 短時間のネットワークスキャンで受信候補を列挙 |

### 例

```json
{
  "cmd": "RTP_START_SESSION",
  "params": {
    "session_id": "aes67-main",
    "bind_address": "0.0.0.0",
    "port": 6000,
    "sample_rate": 48000,
    "payload_type": 97,
    "multicast": true,
    "multicast_group": "239.69.0.1",
    "interface": "eth0",
    "bits_per_sample": 24,
    "target_latency_ms": 5
  }
}
```

成功レスポンス:

```json
{
  "status": "ok",
  "message": "RTP session started",
  "data": {
    "session_id": "aes67-main",
    "port": 6000,
    "sample_rate": 48000
  }
}
```

`RTP_LIST_SESSIONS` は `sessions` 配列を返し、各要素には以下が含まれる:

- `packets_received`, `packets_dropped`
- `ssrc`, `last_rtp_timestamp`
- `ptp_locked`, `ptp_offset_ns`
- `last_packet_unix_ms`

## Config 連携

`config.json` に `rtp` セクションを追加することで自動起動が可能。

```json
"rtp": {
  "enabled": true,
  "autoStart": true,
  "sessionId": "aes67-main",
  "bindAddress": "0.0.0.0",
  "port": 6000,
  "payloadType": 97,
  "sampleRate": 48000,
  "channels": 2,
  "bitsPerSample": 24,
  "multicast": true,
  "multicastGroup": "239.69.0.1"
}
```

`autoStart` が `true` の場合、daemon 起動時に `RtpSessionManager` が初期化され、
PipeWire 入力と並列で PCM を供給できる。ZeroMQ コマンドは常に利用可能で、
制御プレーンから複数セッションのライフサイクルを管理できる。

## Control Plane API (Issue #359)

FastAPI (`web/main.py`) から RTP セッションを管理するための REST API を追加した。

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rtp/sessions` | SessionConfig を受け取り `RTP_START_SESSION` を送信 |
| `GET` | `/api/rtp/sessions` | バックグラウンドポーラがキャッシュしたテレメトリ一覧を返す |
| `GET` | `/api/rtp/sessions/{id}` | 単一セッションの最新メトリクス (`RTP_GET_SESSION`) |
| `DELETE` | `/api/rtp/sessions/{id}` | `RTP_STOP_SESSION` を送信し停止 |

### Pydantic モデル

- `RtpEndpointSettings`: `bind_address`, `port`, `source_host`, `multicast_group`
- `RtpFormatSettings`: `sample_rate`, `channels`, `bits_per_sample`, `payload_type`
- `RtpSyncSettings`: `target_latency_ms`, `watchdog_timeout_ms`, `enable_ptp`
- `RtpSecurityConfig`: Optional SRTPキー (`crypto_suite`, `key_base64`)
- `RtpSdpConfig`: 任意の SDP テンプレート（未指定時は AES67 互換で自動生成）。`a=rtpmap` に
  `L16/48000/2` などが含まれていれば `payload_type` / `sample_rate` / `channels` /
  `bits_per_sample` を自動で上書きする（手入力ミス防止）。
  Multi-rate が有効な場合は `sample_rate` に合わせてアップサンプラ入力レートも自動切り替えする。無効時は警告のみ。

API リクエスト例:

```json
POST /api/rtp/sessions
{
  "session_id": "aes67-main",
  "endpoint": {
    "bind_address": "0.0.0.0",
    "port": 6000,
    "multicast": true,
    "multicast_group": "239.69.0.1"
  },
  "format": { "sample_rate": 48000, "channels": 2, "bits_per_sample": 24 },
  "sync": { "target_latency_ms": 5, "telemetry_interval_ms": 1000 },
  "rtcp": { "enable": true },
  "security": { "crypto_suite": "AES_CM_128_HMAC_SHA1_80", "key_base64": "kK1B..." }
}
```

### バックグラウンドテレメトリ

`web/services/rtp.py` に `RtpTelemetryPoller` を実装し、1.5s間隔で `RTP_LIST_SESSIONS` をポーリングする。
結果は `RtpTelemetryStore` にキャッシュされ `GET /api/rtp/sessions` から即時取得できる。
`MAGICBOX_DISABLE_RTP_POLLING=1` を指定するとポーリングを無効化（テスト用途）。

### Discovery Scanner (#372)

`RTP_DISCOVER_STREAMS` コマンドは `rtp.discovery` ブロックで挙動を制御する。
デフォルト例:

```json
"rtp": {
  "...": "...",
  "discovery": {
    "scanDurationMs": 250,
    "cooldownMs": 1500,
    "maxStreams": 12,
    "enableMulticast": true,
    "enableUnicast": true,
    "ports": [5004, 6000]
  }
}
```

- `scanDurationMs` (50-5000ms): UDPソケットをバインドしてRTPパケットを待ち受ける時間
- `cooldownMs` (>=250ms): 直近のスキャン結果をキャッシュする最小インターバル
- `maxStreams` (1-64): レスポンス内で返却する候補数の上限
- `enableMulticast` / `enableUnicast`: それぞれのアドレスタイプを検出対象に含めるか
- `ports`: スキャン対象のポートリスト。`rtp.port` と AES67 デフォルト(5004)は自動追加される

レスポンスは `streams` 配列に `session_id` / `display_name` / `source_host` /
`port` / `payload_type` / `sample_rate` / `existing_session` などのヒントを含み、UI や REST API
のプリセット生成に利用できる。

## Web UI (Issue #360)

Issue #360 では Control Plane API を直接叩けないメンバー向けに `/rtp` ページを追加した。FastAPI 側から配信されるプレーンHTML/JSで以下を提供する:

- **Session Builder**: SDP貼り付け、バインドIP/ポート、同期モード（低遅延/安定/PTP）、SRTPキー入力。即時バリデーションと進捗インジケータを備え、`POST /api/rtp/sessions` をラップ。
- **Telemetry Board**: `GET /api/rtp/sessions` のキャッシュをカード化。接続状態、RTCP遅延、ネットワークジッタ、PTPロックの有無、遅延パケット数を視覚表示。停止ボタンから `DELETE /api/rtp/sessions/{id}` を発行。
- **UX 保護**: SRTP有効時のみBase64必須、PTPモード選択時のみインターフェース必須、トースト通知 & 自動ポーリング（5s）。

このページは `/`（ユーザーページ）や `/admin` と同じく FastAPI 組み込みテンプレート (`web/templates/rtp.py`) で提供する。Storybook 等は未導入だが `tests/python/test_rtp_ui.py` で必須DOM要素とJSフックをカバーしている。

### cURL での検証手順

```bash
uv sync
uv run uvicorn web.main:app --reload --port 8000

curl -X POST http://127.0.0.1:8000/api/rtp/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id":"aes67-main","endpoint":{"port":6000},"format":{"sample_rate":48000}}'

curl http://127.0.0.1:8000/api/rtp/sessions
curl http://127.0.0.1:8000/api/rtp/sessions/aes67-main
curl -X DELETE http://127.0.0.1:8000/api/rtp/sessions/aes67-main
```

この手順を README にも掲載し、受け入れ条件「API を叩きセッション開始／停止を確認できる」に対応した。
