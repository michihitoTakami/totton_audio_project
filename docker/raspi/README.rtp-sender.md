# RTP Sender (RasPi Simulation)

このディレクトリには、RasPiをシミュレートしてPCのPipeWire RTPストリームをMagic Boxに中継するDocker環境が含まれています。

## 概要

```
[PC PipeWire] --RTP--> [Docker: rtp-sender] --RTP+SDP--> [Magic Box]
     ↓                      ↓                              ↓
224.0.0.56:46000     マルチキャスト受信         RtpSessionManager
                     フォーマット検出
                     SDP自動生成・送信
```

## 前提条件

1. **PipeWireがRTP送信中**
   ```bash
   # 現在の設定確認
   pactl list sinks | grep -A 10 "RTP Network Sink"
   ```
   - マルチキャストグループ: `224.0.0.56`
   - ポート: `46000`
   - フォーマット: `s16be 2ch 44100Hz` (または任意)
   - **設定方法**: `docs/setup/raspi/pipewire_rtp_sender.md` 参照

2. **Magic Boxが稼働中**
   - IPアドレス: `192.168.1.10` (要変更)
   - REST API: `http://192.168.1.10:8000`

## セットアップ

### 1. Magic BoxのIPアドレス設定

`docker/raspi/docker-compose.raspi-sender.yml` を編集:

```yaml
environment:
  - MAGIC_BOX_HOST=192.168.1.10  # ← 実際のMagic BoxのIPに変更
```

### 2. Docker Composeで起動

```bash
cd docker/raspi
docker compose -f docker-compose.raspi-sender.yml up -d --build
```

### 3. ログ確認

```bash
# リアルタイムログ
docker logs -f raspi-rtp-sender

# ファイルから確認
tail -f logs/rtp-sender/rtp_sender.log
```

## 動作確認

### 1. RTP受信確認

```bash
docker logs raspi-rtp-sender | grep "Detected format"
```

期待される出力:
```
[INFO] Detected format: 44100Hz, 2ch, 16bit, PT127
[INFO] Successfully registered RTP session: 44100Hz, 2ch, 16bit
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

### フォーマット検出が失敗する

ログに以下が出ない場合:
```
[INFO] Detected format: ...
```

PipeWireの送信を確認:
```bash
pactl list sinks | grep -A 5 "RTP Network Sink"
wpctl status | grep "RTP"
```

## 環境変数一覧

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `RTP_RECV_MULTICAST_GROUP` | `224.0.0.56` | PipeWireのマルチキャストグループ |
| `RTP_RECV_PORT` | `46000` | 受信ポート |
| `MAGIC_BOX_HOST` | `192.168.1.10` | Magic BoxのIPアドレス |
| `MAGIC_BOX_API_PORT` | `8000` | Magic BoxのREST APIポート |
| `RTP_SEND_PORT` | `46001` | Magic Boxへの送信ポート |
| `AUTO_REGISTER` | `true` | 起動時に自動登録 |
| `SEND_SDP_ON_RATE_CHANGE` | `true` | レート変更時にSDP再送信 |
| `LOG_LEVEL` | `INFO` | ログレベル (DEBUG/INFO/WARNING/ERROR) |

## クリーンアップ

```bash
# コンテナ停止・削除
cd docker/raspi
docker compose -f docker-compose.raspi-sender.yml down

# イメージ削除
docker rmi raspi-rtp-sender:latest

# ログ削除
rm -rf logs/rtp-sender
```

## Docker rtp-senderとの連携

1. PipeWireでRTP送信開始（`docs/setup/raspi/pipewire_rtp_sender.md` の設定）
2. Docker rtp-senderが自動受信＆Magic Boxに転送

```bash
# Docker起動
cd docker/raspi
docker compose -f docker-compose.raspi-sender.yml up -d

# ログ確認（フォーマット検出されるはず）
docker logs -f raspi-rtp-sender
```

## 本番RasPi展開

本番環境では、このDockerコンテナの代わりに以下を使用:

1. **systemdサービス**: `scripts/raspi/rtp_sender.py` を直接実行
2. **PipeWire設定**: UAC2デバイスから直接RTP送信
3. **ネットワーク設定**: 静的IPで閉域LAN構築

詳細は `docs/setup/raspi/` 配下のドキュメントを参照。
