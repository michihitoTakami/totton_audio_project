# RTP Sender (RasPi Simulation)

このディレクトリには、RasPiをシミュレートしてPCのPipeWire RTPストリームをMagic Boxに中継するDocker環境が含まれています。

## 概要

```
[PC PipeWire] --RTP--> [Docker: rtp-sender] --RTP+SDP--> [Magic Box]
     ↓                      ↓                              ↓
224.0.0.56:46000      パケット転送              RtpSessionManager
                      (環境変数でフォーマット指定)
```

## 設計方針

**重要**: RTPパケットからオーディオフォーマット（サンプルレート、チャンネル数、ビット深度）を推定することは原理的に不可能です。そのため、このrtp-senderは以下の方針で設計されています：

1. **フォーマットは環境変数で明示的に指定** - PipeWireの設定と同じ値を指定
2. **起動時にSDPを送信** - Magic BoxのREST APIにフォーマット情報を送信
3. **RTPパケットはそのまま転送** - 受信したパケットに対する解析・変更は行わない

## 前提条件

1. **PipeWireがRTP送信中**
   ```bash
   # 現在の設定確認
   pactl list sinks | grep -A 10 "RTP Network Sink"
   ```
   - マルチキャストグループ: `224.0.0.56`
   - ポート: `46000`
   - フォーマット: PipeWire設定に依存
   - **設定方法**: `docs/setup/configs/` 配下のサンプル設定参照

2. **Magic Boxが稼働中**
   - IPアドレス: `192.168.1.10` (要変更)
   - REST API: `http://192.168.1.10:8000`

## セットアップ

### 1. 環境変数をPipeWire設定と一致させる

`docker-compose.yml` を編集して、PipeWireの `rtp-sink.conf` と同じ値を設定:

```yaml
environment:
  # ⚠️ PipeWireのrtp-sink.conf設定と一致させること！
  - RTP_SAMPLE_RATE=44100    # audio.rate と一致
  - RTP_CHANNELS=2           # audio.channels と一致
  - RTP_BITS_PER_SAMPLE=16   # format: S16=16, S24=24, S32=32
  - RTP_PAYLOAD_TYPE=127     # 動的ペイロードタイプ

  # Magic Box接続設定
  - MAGIC_BOX_HOST=192.168.1.10  # ← 実際のMagic BoxのIPに変更
```

### 2. Docker Composeで起動

```bash
cd docker/local/raspi-simulation
docker compose up -d --build
```

### 3. ログ確認

```bash
# リアルタイムログ
docker logs -f raspi-rtp-sender

# ファイルから確認
tail -f logs/rtp-sender/rtp_sender.log
```

## 動的レート変更テスト

PipeWire側のレートを変更してテストする場合：

### 手順

1. PipeWireの設定を変更（例: 44100Hz → 96000Hz）
   ```bash
   vim ~/.config/pipewire/pipewire.conf.d/rtp-sink.conf
   systemctl --user restart pipewire
   ```

2. docker-compose.ymlの環境変数を更新
   ```bash
   # RTP_SAMPLE_RATE=96000 に変更
   vim docker-compose.yml
   ```

3. コンテナを再作成
   ```bash
   docker compose up -d
   ```

4. ログで確認
   ```bash
   docker logs raspi-rtp-sender | grep "Successfully registered"
   # [INFO] Successfully registered RTP session: 96000Hz, 2ch, 16bit, PT127
   ```

### SDP再送信（上級者向け）

SIGHUPシグナルでSDPを再送信できます（環境変数は変わらない）：
```bash
docker kill -s HUP raspi-rtp-sender
```

これはMagic Box側でセッションを再作成したい場合に使用します。

## 動作確認

### 1. SDP送信確認

```bash
docker logs raspi-rtp-sender | grep "Successfully registered"
```

期待される出力:
```
[INFO] Successfully registered RTP session: 44100Hz, 2ch, 16bit, PT127
```

### 2. Magic Box側でセッション確認

```bash
curl http://192.168.1.10:8000/api/rtp/sessions
```

期待される出力:
```json
{
  "sessions": [
    {
      "session_id": "raspi-uac2",
      "sample_rate": 44100,
      "channels": 2,
      "packets_received": 1234,
      ...
    }
  ]
}
```

## 環境変数一覧

### 受信設定（PC PipeWireから）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `RTP_RECV_MULTICAST_GROUP` | `224.0.0.56` | PipeWireのマルチキャストグループ |
| `RTP_RECV_PORT` | `46000` | 受信ポート |

### 送信設定（Magic Boxへ）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `MAGIC_BOX_HOST` | `192.168.1.10` | Magic BoxのIPアドレス |
| `MAGIC_BOX_API_PORT` | `8000` | Magic BoxのREST APIポート |
| `RTP_SEND_PORT` | `46001` | Magic Boxへの送信ポート |

### オーディオフォーマット設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `RTP_SAMPLE_RATE` | `44100` | サンプルレート（Hz） |
| `RTP_CHANNELS` | `2` | チャンネル数 |
| `RTP_BITS_PER_SAMPLE` | `16` | ビット深度（16/24/32） |
| `RTP_PAYLOAD_TYPE` | `127` | RTPペイロードタイプ |

### セッション設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `RTP_SESSION_ID` | `raspi-uac2` | セッション識別子 |
| `LOG_LEVEL` | `INFO` | ログレベル (DEBUG/INFO/WARNING/ERROR) |

## トラブルシューティング

### マルチキャストが受信できない

```bash
# ファイアウォール無効化（テスト用）
sudo ufw disable

# マルチキャストルート確認
ip route | grep 224
```

### Magic Box APIに接続できない

```bash
# Magic Boxへの疎通確認
docker exec raspi-rtp-sender curl -v http://192.168.1.10:8000/status
```

### フォーマット不一致で音が出ない

SDPのフォーマット情報がPipeWire設定と異なると、Magic Box側で正しくデコードできません。

1. PipeWireの設定を確認:
   ```bash
   cat ~/.config/pipewire/pipewire.conf.d/rtp-sink.conf
   ```

2. docker-compose.ymlの環境変数と比較して一致させる

## クリーンアップ

```bash
# コンテナ停止・削除
cd docker/local/raspi-simulation
docker compose down

# イメージ削除
docker rmi raspi-rtp-sender:latest

# ログ削除
rm -rf logs/rtp-sender
```

## 本番RasPi展開

本番環境では、このDockerコンテナの代わりに以下を使用:

1. **systemdサービス**: `scripts/raspi/rtp_sender.py` を直接実行
2. **PipeWire設定**: UAC2デバイスから直接RTP送信
3. **ネットワーク設定**: 静的IPで閉域LAN構築

詳細は `docs/setup/raspi/` 配下のドキュメントを参照。
