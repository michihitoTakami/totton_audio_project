# PipeWire RTP Sender 設定ガイド

このドキュメントでは、ローカルPC（RasPi想定）でPipeWireからRTPストリームを送信する設定を説明します。

## 概要

PipeWireの`libpipewire-module-rtp-sink`モジュールを使用して、PCの音声出力をRTP over UDP/IPで送信します。

```
[Audio Source (Spotify, Chrome, etc)]
    ↓
[PipeWire RTP Sink]
    ↓
[UDP Multicast: 224.0.0.56:46000]
    ↓
[Magic Box or Docker rtp-sender]
```

## セットアップ手順

### 1. 設定ファイルの配置

PipeWireユーザー設定ディレクトリに設定ファイルを作成します。

```bash
# 設定ディレクトリ作成
mkdir -p ~/.config/pipewire/pipewire.conf.d

# 設定ファイルをコピー
cp docs/setup/raspi/configs/rtp-sink.conf ~/.config/pipewire/pipewire.conf.d/
```

### 2. PipeWireの再起動

```bash
# PipeWire を再起動（セッション全体）
systemctl --user restart pipewire pipewire-pulse

# または、一度ログアウト/ログインする
```

### 3. RTPシンクの確認

```bash
# シンクが作成されているか確認
pactl list sinks short | grep rtp

# 期待される出力:
# 39  rtp-sink  PipeWire  s16be 2ch 44100Hz  RUNNING
```

### 4. RTPシンクをデフォルトに設定

```bash
# デフォルトシンクに設定
pactl set-default-sink rtp-sink

# 現在のデフォルトを確認
pactl info | grep "Default Sink"
```

### 5. 音声ストリームの確認

```bash
# RTPパケットが送信されているか確認
sudo tcpdump -i any -n udp port 46000

# 期待される出力（音声再生中）:
# IP 192.168.1.100.xxxxx > 224.0.0.56.46000: UDP, length 1280
```

## 設定ファイルの詳細

### 基本設定（44.1kHz, 16bit, ステレオ）

```conf
# ~/.config/pipewire/pipewire.conf.d/rtp-sink.conf
context.modules = [
    {   name = libpipewire-module-rtp-sink
        args = {
            local.ip = "0.0.0.0"
            destination.ip = "224.0.0.56"
            destination.port = 46000

            format = "S16"          # 16-bit signed integer
            audio.rate = 44100      # Sample rate
            audio.channels = 2      # Stereo

            mtu = 1280
            ttl = 32

            stream.props = {
                node.name = "rtp-sink"
                node.description = "RTP Network Sink"
                media.class = "Audio/Sink"
            }
        }
    }
]
```

### 高品質設定（96kHz, 24bit, ステレオ）

98kHzをテストする場合は、まず96kHzで試すことを推奨：

```conf
context.modules = [
    {   name = libpipewire-module-rtp-sink
        args = {
            local.ip = "0.0.0.0"
            destination.ip = "224.0.0.56"
            destination.port = 46000

            format = "S24"          # 24-bit signed integer
            audio.rate = 96000      # 96kHz sample rate
            audio.channels = 2

            mtu = 1280
            ttl = 32

            stream.props = {
                node.name = "rtp-sink-hifi"
                node.description = "RTP Network Sink (Hi-Fi)"
                media.class = "Audio/Sink"
            }
        }
    }
]
```

### ユニキャスト送信（Magic Box直接送信）

マルチキャストが使えない環境では、ユニキャスト送信を使用：

```conf
context.modules = [
    {   name = libpipewire-module-rtp-sink
        args = {
            local.ip = "0.0.0.0"
            destination.ip = "192.168.1.10"  # Magic BoxのIPアドレス
            destination.port = 46000

            format = "S16"
            audio.rate = 44100
            audio.channels = 2

            mtu = 1280
            ttl = 32  # ユニキャストでも指定可能（意味はない）

            stream.props = {
                node.name = "rtp-sink"
                node.description = "RTP Network Sink (Unicast to Magic Box)"
                media.class = "Audio/Sink"
            }
        }
    }
]
```

## パラメータリファレンス

| パラメータ | 説明 | デフォルト | 推奨値 |
|-----------|------|-----------|--------|
| `local.ip` | 送信元IPアドレス | `0.0.0.0` | `0.0.0.0` (全インターフェース) |
| `destination.ip` | 送信先IPアドレス | - | `224.0.0.56` (マルチキャスト) |
| `destination.port` | 送信先ポート | `5004` | `46000` (Magic Box待受ポート) |
| `format` | 音声フォーマット | `S16` | `S16` / `S24` / `S32` |
| `audio.rate` | サンプリングレート | `48000` | `44100` / `48000` / `96000` |
| `audio.channels` | チャンネル数 | `2` | `2` (ステレオ) |
| `mtu` | 最大パケットサイズ | `1280` | `1280` (安全値) |
| `ttl` | Time-To-Live | `1` | `32` (ローカルネットワーク) |

### フォーマット一覧

| `format` | ビット深度 | バイトオーダー | 説明 |
|----------|-----------|---------------|------|
| `S16` | 16bit | Big Endian | L16, CD品質 |
| `S24` | 24bit | Big Endian | L24, 高品質 |
| `S32` | 32bit | Big Endian | L32, 最高品質 |

**注意**: PipeWire RTPモジュールは自動的にBig Endianで送信します（RTP標準）。

## トラブルシューティング

### RTPシンクが表示されない

```bash
# PipeWireのログを確認
journalctl --user -u pipewire -n 50

# 設定ファイルの文法エラーを確認
pipewire -c ~/.config/pipewire/pipewire.conf --check
```

### マルチキャストが届かない

```bash
# マルチキャストルーティング確認
ip route | grep 224

# なければ追加
sudo ip route add 224.0.0.0/4 dev eth0  # eth0 を実際のインターフェース名に変更

# ファイアウォール確認
sudo ufw status
sudo ufw allow out to 224.0.0.0/4
```

### 音が出ない

```bash
# RTPシンクに音声ストリームが流れているか確認
pactl list sink-inputs

# 強制的に既存のストリームをRTPシンクに移動
pactl list short sink-inputs | awk '{print $1}' | xargs -I{} pactl move-sink-input {} rtp-sink
```

### レイテンシが高い

`quantum`（バッファサイズ）を調整：

```conf
stream.props = {
    node.name = "rtp-sink"
    node.description = "RTP Network Sink"
    media.class = "Audio/Sink"
    node.latency = "128/44100"  # 128サンプル ≈ 2.9ms
}
```

## Docker rtp-senderとの連携

1. PipeWireでRTP送信開始（このドキュメントの設定）
2. Docker rtp-senderが自動受信＆Magic Boxに転送

```bash
# Docker起動
cd docker
docker compose -f docker-compose.raspi-sender.yml up -d

# ログ確認（フォーマット検出されるはず）
docker logs -f raspi-rtp-sender
```

## 本番RasPi展開時の注意

本番RasPiでは、以下の点が異なります：

1. **UAC2デバイスからの入力**
   ```conf
   # RasPiではrtp-sinkではなく、UAC2からの入力を直接RTP化
   context.modules = [
       {   name = libpipewire-module-rtp-sink
           args = {
               destination.ip = "192.168.1.10"  # Magic Box
               destination.port = 46000

               # UAC2デバイスを監視
               stream.props = {
                   node.target = "alsa_input.usb-*"  # UAC2デバイス
               }
           }
       }
   ]
   ```

2. **自動起動設定**
   ```bash
   # systemdで自動起動
   systemctl --user enable pipewire pipewire-pulse
   ```

3. **ネットワーク設定**
   - 静的IP設定（例: 192.168.1.5）
   - 閉域LAN構築（Magic Boxと直結）
