# RasPiシミュレーション環境（ローカル開発）

このドキュメントでは、ローカルPC上でRasPi UAC2 → Magic Box構成をシミュレートする開発環境について説明します。

## 概要

本番RasPi環境では、UAC2デバイスとして受信したPCM音声をRTP経由でMagic Boxに送信します。
この開発環境では、PipeWireのRTP送信機能を使ってその動作をシミュレートします。

```
[PC PipeWire] → RTP (224.0.0.56:46000)
    ↓
[Docker: RTP Sender] 環境変数指定 → SDP送信
    ↓
[Magic Box] RtpSessionManager → GPU処理
```

## 構成要素

### 1. PipeWire RTP Sink
- PC上の音声をRTPマルチキャストで送信
- 設定: `docs/setup/raspi/pipewire_rtp_sender.md`

### 2. RTP Senderコンテナ
- RTPストリームを受信しMagic Boxに転送
- PipeWire設定と一致するフォーマットを環境変数で指定
- 起動時（およびSIGHUP受信時）にSDPをMagic Box REST APIへ送信
- RTPパケットを解析せずそのまま転送

### 3. Magic Box（オプション）
- 同じPC上で動かす場合は別途起動
- 別マシンで動かす場合はそちらを使用

## セットアップ

### 1. PipeWire RTP送信設定

```bash
# 設定ファイルをコピー
mkdir -p ~/.config/pipewire/pipewire.conf.d
cp docs/setup/raspi/configs/rtp-sink.conf ~/.config/pipewire/pipewire.conf.d/

# PipeWire再起動
systemctl --user restart pipewire pipewire-pulse

# RTPシンク確認
pactl list sinks short | grep rtp

# デフォルトシンクに設定
pactl set-default-sink rtp-sink
```

詳細: `docs/setup/raspi/pipewire_rtp_sender.md`

### 2. Magic BoxのIPアドレス設定

`docker/local/raspi-simulation/docker-compose.yml` を編集:

```yaml
environment:
  - MAGIC_BOX_HOST=192.168.1.10  # ← 実際のMagic BoxのIPに変更
```

同じPC上でMagic Boxも動かす場合は `localhost` または `127.0.0.1` を指定。

### 3. RTP Senderコンテナ起動

```bash
cd docker/local/raspi-simulation
docker compose up -d --build
```

### 4. ログ確認

```bash
# リアルタイムログ
docker logs -f raspi-rtp-sender

# 期待される出力:
# [INFO] Successfully registered RTP session: 44100Hz, 2ch, 16bit, PT127
# [INFO] Forwarding RTP packets...
```

## 動作確認

### 1. SDP送信確認

```bash
docker logs raspi-rtp-sender | grep "Successfully registered"
```

### 2. RTP転送確認

```bash
docker logs raspi-rtp-sender | grep "Forwarding RTP packets"
```

### 3. Magic Box側でセッション確認

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

# なければ追加
sudo ip route add 224.0.0.0/4 dev eth0  # eth0 を実際のインターフェース名に変更
```

### Magic Box APIに接続できない

```bash
# Magic Boxへの疎通確認
docker exec raspi-rtp-sender curl -v http://192.168.1.10:8000/status
```

### フォーマット設定が一致しない

PipeWireの送信設定と `docker/local/raspi-simulation/docker-compose.yml` に記載した `RTP_*` 環境変数が一致しないとMagic Box側で音が出ません。

1. PipeWireの設定を確認:
   ```bash
   cat ~/.config/pipewire/pipewire.conf.d/rtp-sink.conf
   ```
2. docker-compose.ymlの環境変数を再確認:
   ```bash
   grep RTP_ docker/local/raspi-simulation/docker-compose.yml
   ```
3. 値を揃えたらコンテナを再作成するか、SIGHUPでSDPを再送信:
   ```bash
   cd docker/local/raspi-simulation
   docker compose up -d
   # もしくは再送信のみ
   docker kill -s HUP raspi-rtp-sender
   ```

## 同じPC上でMagic Boxも動かす場合

### 1. Magic Boxを起動

```bash
cd docker/local/pipewire
docker compose up -d --build
```

### 2. docker-compose.ymlを編集

```yaml
environment:
  - MAGIC_BOX_HOST=host.docker.internal  # Dockerホスト内のMagic Boxにアクセス
  - MAGIC_BOX_API_PORT=80
```

または、`network_mode: host` を使用してホストネットワークで動かす。

## 本番RasPi環境との違い

| 項目 | ローカルシミュレーション | 本番RasPi |
|------|---------------------|----------|
| 入力 | PipeWire（任意ソース） | USB UAC2デバイス |
| RTP送信 | PipeWire module-rtp-sink | PipeWire + UAC2入力 |
| 中継 | Docker（開発用） | systemdサービス |
| ネットワーク | 同一マシン or LAN | 閉域LAN（直結） |
| Magic Box | Docker or 別マシン | Jetson Orin Nano |

## 関連ドキュメント

- `docs/setup/raspi/pipewire_rtp_sender.md`: PipeWire RTP送信設定
- `docker/local/raspi-simulation/README.md`: Docker環境の詳細
- `docker/raspi/README.md`: 本番RasPi展開
