# RTPダイレクト受信モード

## 概要

RTPダイレクト受信モードは、PipeWire RTPレシーバーをバイパスし、PCからのRTPストリームをC++デーモン内で直接受信・処理する機能です。

### メリット
- **低レイテンシ**: PipeWireのオーバーヘッドを回避
- **シンプルなルーティング**: PC → UDP/RTP → GPUデーモン → DAC
- **パケットロスの可視化**: RTPセッションメトリクスで詳細な統計情報を取得可能

## デフォルト設定

### RTPセッション設定 (config.json)

```json
{
  "rtp": {
    "enabled": true,
    "autoStart": true,
    "sessionId": "pc_stream",
    "bindAddress": "0.0.0.0",
    "port": 46000,
    "payloadType": 127,
    "sampleRate": 44100,
    "channels": 2,
    "bitsPerSample": 16,
    "bigEndian": true,
    "signedSamples": true,
    "targetLatencyMs": 5,
    "watchdogTimeoutMs": 500,
    "enableRtcp": true
  }
}
```

### 主要パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `port` | 46000 | 受信ポート（PipeWire RTP送信と一致させる） |
| `payloadType` | 127 | RTPペイロードタイプ（PipeWireのデフォルト） |
| `sampleRate` | 44100 | サンプリングレート（44.1kHz系） |
| `bitsPerSample` | 16 | ビット深度（S16BE） |
| `bigEndian` | true | ビッグエンディアン形式 |
| `targetLatencyMs` | 5 | ターゲットレイテンシ（低遅延） |

## PC側（RTP送信元）の設定

### PipeWire RTP送信モジュール

PipeWireでRTPストリームを送信するには、`module-rtp-send`を使用します：

```bash
# RTP送信モジュールをロード（例：ポート46000、Payload Type 127）
pactl load-module module-rtp-send \
  destination_ip=192.168.11.x \
  port=46000 \
  payload=127 \
  source=<source_name>
```

**注意**: `destination_ip`はJetson（Magic Box）のIPアドレスに変更してください。

## モード切り替え

### RTPモード → PipeWireモードに戻す

1. **config.jsonを編集**

```json
{
  "rtp": {
    "enabled": false,
    "autoStart": false,
    ...
  }
}
```

2. **デーモンを再起動**

```bash
# WEBコンソールから: 停止 → 起動
# またはCLI:
./scripts/daemon.sh restart
```

3. **確認**

デーモン起動時に以下のメッセージが表示されます：
```
Creating PipeWire input (capturing from gpu_upsampler_sink)...
```

### PipeWireモード → RTPモードに切り替える

1. **config.jsonを編集**（上記のデフォルト設定を参照）
2. **デーモンを再起動**
3. **確認**

RTPセッションが自動起動されます：
```bash
# ZeroMQ経由で確認
uv run python -c "
import zmq, json
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('ipc:///tmp/gpu_os.sock')
sock.send_json({'cmd': 'RTP_LIST_SESSIONS'})
print(json.dumps(sock.recv_json(), indent=2))
sock.close()
"
```

## WEB管理画面

### ユーザーページ (http://localhost:8000/)

1. **RTPスキャン**: 「RTP入力をスキャン」ボタンでネットワーク上のRTPストリームを検索
2. **候補表示**: スキャン結果がドロップダウンに表示される
3. **稼働中セッション**: 現在受信中のセッション情報（パケット数、レート等）

### 表示される情報

- **ストリーム候補**: `192.168.11.15:59764 (PT127)`
- **稼働中セッション**: `pc_stream: 20,000+ packets received`
- **詳細情報**: サンプルレート、チャンネル数、Payload Type

## トラブルシューティング

### スキャンで候補が見つからない

1. **PC側のRTP送信を確認**
```bash
# PipeWireモジュールがロードされているか確認
pactl list modules | grep rtp-send
```

2. **ネットワーク接続を確認**
```bash
# Jetson側でパケットが届いているか確認
ss -ulnp | grep 46000
```

3. **ファイアウォール設定を確認**
```bash
# UDPポート46000が開いているか確認
sudo ufw status
```

### パケットが受信されない（packets_received: 0）

**Payload Typeの不一致**が最も一般的な原因です：

```bash
# PC側のPayload Typeを確認
pw-cli info <rtp-source-id> | grep rtp.payload

# Jetson側の設定と一致させる
# config.json: "payloadType": 127
```

### WEBコンソールからデーモンが起動しない

RTPモード時は`web/services/daemon.py`が自動的にPipeWireチェックをスキップします。ログを確認：

```bash
tail -f /var/log/gpu_upsampler/daemon.log
```

## 参考情報

### オーディオフロー（RTPモード）

```
[PC (RTP送信)]
    ↓ UDP/RTP (ネットワーク経由)
    ↓ ポート 46000, Payload Type 127
[RtpSessionManager] (C++ 内蔵)
    ↓ デコード (S16BE → float)
    ↓ frameCallback
[process_interleaved_block()]
    ↓ GPU処理（16x アップサンプリング）
[ALSA → SMSL DAC]
```

### 関連ファイル

- `config.json` - RTP設定
- `src/rtp_session_manager.cpp` - RTP受信エンジン
- `src/alsa_daemon.cpp` - 統合ポイント
- `web/services/daemon.py` - WEB起動制御
- `web/templates/user.py` - ユーザーUI
