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

