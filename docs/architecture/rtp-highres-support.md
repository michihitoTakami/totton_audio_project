# RTPハイレゾ対応アーキテクチャ

## 概要

Magic Boxは、Raspberry Pi 5からJetson Orin NanoへのRTP転送において、**完全なハイレゾ透過性**を実現しています。

入力されたオーディオフォーマット（サンプルレート、ビット深度、チャンネル数）は、RaspberryPi側でSDP（Session Description Protocol）として記述され、Jetson側で自動的に認識・適用されます。

## サポートフォーマット

### サンプルレート
- **44.1kHz系**: 44.1k / 88.2k / 176.4k / 352.8k
- **48kHz系**: 48k / 96k / 192k / 384k / 768k

### ビット深度
- **16-bit**: CD品質
- **24-bit**: ハイレゾ標準（推奨）
- **32-bit**: 最高品質

### チャンネル数
- **2ch**: ステレオ（標準）
- **最大8ch**: マルチチャンネル対応

## データフロー

```
┌─────────────────────────────────────────┐
│      Raspberry Pi 5 (Input Hub)         │
├─────────────────────────────────────────┤
│  入力: USB UAC2 / Spotify / AirPlay等   │
│         ↓                               │
│  PipeWire: サンプルレート検知            │
│         ↓                               │
│  SDP生成:                               │
│    v=0                                  │
│    o=- 0 0 IN IP4 192.168.1.5          │
│    s=RaspberryPi Audio                 │
│    c=IN IP4 192.168.1.10               │
│    t=0 0                               │
│    m=audio 46000 RTP/AVP 96            │
│    a=rtpmap:96 L24/96000/2  ← ここ！   │
│         ↓                               │
│  RTP送信 (L24形式、96kHz/2ch)           │
└─────────────────────────────────────────┘
             ↓ Ethernet
┌─────────────────────────────────────────┐
│     Jetson Orin Nano (Processing)       │
├─────────────────────────────────────────┤
│  RTP受信                                │
│         ↓                               │
│  SDP自動パース:                         │
│    - payloadType: 96                   │
│    - sampleRate: 96000                 │
│    - channels: 2                       │
│    - bitsPerSample: 24                 │
│         ↓                               │
│  GPU Upsamplerへレート通知              │
│         ↓                               │
│  switchToInputRate(96000)              │
│    → filter_48k_8x_640k_min_phase.bin │
│         ↓                               │
│  GPU処理: 96kHz → 768kHz (8倍)         │
│         ↓                               │
│  DAC出力: 768kHz/32bit                 │
└─────────────────────────────────────────┘
```

## SDP自動パース機能

### 実装箇所
`src/rtp_session_manager.cpp:parseRtpmapLine()` (113-173行)

### 対応フォーマット
```sdp
a=rtpmap:<PT> L<bits>/<rate>/<channels>
```

**例:**
```sdp
a=rtpmap:96 L24/96000/2    # 96kHz, 24-bit, Stereo
a=rtpmap:97 L16/44100/2    # 44.1kHz, 16-bit, Stereo
a=rtpmap:98 L32/192000/2   # 192kHz, 32-bit, Stereo
```

### パース処理フロー

1. **SDPを行単位で解析**
   ```cpp
   std::stringstream ss(config.sdp);
   std::string line;
   while (std::getline(ss, line)) {
       // a=rtpmap: で始まる行を探す
   }
   ```

2. **rtpmap行の分解**
   ```
   a=rtpmap:96 L24/96000/2
            ↓
   payloadType = 96
   encoding = "L24/96000/2"
   ```

3. **エンコーディング解析**
   ```
   L24/96000/2
   ↓
   bitsPerSample = 24
   sampleRate = 96000
   channels = 2
   ```

4. **SessionConfigへ適用**
   ```cpp
   config.payloadType = 96;
   config.sampleRate = 96000;
   config.channels = 2;
   config.bitsPerSample = 24;
   ```

## 動的レート切り替え

### トリガー
RTPストリーム内のサンプルレートが変化した場合（例: Spotify 44.1kHz → Roon 192kHz）

### 処理フロー

1. **レート変化検出**
   ```cpp
   // src/alsa_daemon.cpp:handle_rate_change()
   if (detected_sample_rate != g_current_input_rate.load()) {
       // レート変更処理開始
   }
   ```

2. **Soft Mute（Fade-out）**
   ```cpp
   g_soft_mute->startFadeOut();
   // 約50ms かけて音量を0にフェード
   ```

3. **GPU フィルタ切り替え**
   ```cpp
   g_upsampler->switchToInputRate(detected_sample_rate);
   // 例: 96kHz → filter_48k_8x_640k_min_phase.bin
   ```

4. **ストリーミングバッファ再初期化**
   ```cpp
   g_upsampler->initializeStreaming();
   g_stream_input_left.clear();
   g_stream_input_right.clear();
   ```

5. **Soft Mute（Fade-in）**
   ```cpp
   g_soft_mute->startFadeIn();
   // 約50ms かけて元の音量に復帰
   ```

### グリッチフリー保証
- **Total Mute Time**: 約100ms（fade-out 50ms + fade-in 50ms）
- **切り替え中のノイズ**: 完全に抑制
- **音楽再生への影響**: 曲間での切り替えなら気づかない

## GPU Upsamplerとの連携

### マルチレート対応

Jetson側のGPU Upsamplerは、**全8入力レート**に対応しています。

| Input Rate | Rate Family | Filter File | Output Rate |
|-----------|-------------|-------------|-------------|
| 44.1k | 44k | `filter_44k_16x_640k_min_phase.bin` | 705.6k |
| 88.2k | 44k | `filter_44k_8x_640k_min_phase.bin` | 705.6k |
| 176.4k | 44k | `filter_44k_4x_640k_min_phase.bin` | 705.6k |
| 352.8k | 44k | `filter_44k_2x_640k_min_phase.bin` | 705.6k |
| 48k | 48k | `filter_48k_16x_640k_min_phase.bin` | 768k |
| 96k | 48k | `filter_48k_8x_640k_min_phase.bin` | 768k |
| 192k | 48k | `filter_48k_4x_640k_min_phase.bin` | 768k |
| 384k | 48k | `filter_48k_2x_640k_min_phase.bin` | 768k |

### 自動フィルタ選択

```cpp
// 96kHz入力の場合
detectRateFamily(96000) → RATE_48K
selectUpsampleRatio(96000, 768000) → 8x
loadFilter("filter_48k_8x_640k_min_phase.bin")
```

## パフォーマンス

### レイテンシ
- **RTP転送遅延**: < 5ms（Gigabit Ethernet）
- **GPU処理**: 約28倍リアルタイム（余裕あり）
- **レート切り替え**: 約100ms（グリッチフリー）

### 帯域使用量
| Format | Bitrate | 備考 |
|--------|---------|------|
| 44.1kHz/16bit/2ch | 1.4 Mbps | CD品質 |
| 96kHz/24bit/2ch | 4.6 Mbps | ハイレゾ標準 |
| 192kHz/24bit/2ch | 9.2 Mbps | DSD相当 |
| 768kHz/32bit/2ch | 49.2 Mbps | 極限品質 |

**推奨ネットワーク**: Gigabit Ethernet（1000 Mbps）

## 実装ファイル

| ファイル | 役割 |
|---------|------|
| `src/rtp_session_manager.cpp` | RTP受信、SDPパース |
| `src/rtp_session_manager.h` | SessionConfig定義 |
| `src/alsa_daemon.cpp` | レート変更ハンドリング、GPU連携 |
| `src/convolution_engine.cu` | GPU Upsampler（マルチレート対応） |
| `include/convolution_engine.h` | `MULTI_RATE_CONFIGS` 定義 |

## 設定例

### Jetson側（config.json）
```json
{
  "rtp": {
    "enabled": true,
    "autoStart": true,
    "sessionId": "raspi-audio",
    "bindAddress": "0.0.0.0",
    "port": 46000,
    "payloadType": 96,
    "sampleRate": 48000,      // デフォルト値（SDPで上書き）
    "channels": 2,
    "bitsPerSample": 24,
    "bigEndian": true,
    "targetLatencyMs": 5,
    "enableRtcp": true
  }
}
```

**注意**: `sampleRate` はデフォルト値であり、実際の値はSDPで動的に上書きされます。

### Raspberry Pi 5側（docker-compose.yml）
```yaml
services:
  pipewire-rtp:
    image: magicbox/pipewire-rtp-sender
    environment:
      - JETSON_IP=192.168.1.10
      - RTP_PORT=46000
      - PAYLOAD_TYPE=96
      # サンプルレートは入力に応じて自動設定
    network_mode: host
    devices:
      - /dev/snd
```

## トラブルシューティング

### 問題: ハイレゾ音源が44.1kHzにダウンサンプルされる
**原因**: Raspberry Pi側のPipeWire設定が不適切
**対策**:
```bash
# PipeWireの設定を確認
pw-metadata -n settings

# サンプルレート制限を解除
pw-metadata -n settings 0 clock.force-rate 0
```

### 問題: レート切り替え時にノイズが発生
**原因**: Soft Mute機能が無効化されている
**対策**:
```json
// config.json
{
  "softMute": {
    "enabled": true,
    "fadeDurationMs": 50
  }
}
```

### 問題: 768kHz出力ができない
**原因**: DACが768kHzに対応していない
**対策**:
```bash
# DACの対応レートを確認
cat /proc/asound/card*/stream0

# config.jsonで最大レートを制限
{
  "maxOutputRate": 384000  // 384kHzまでに制限
}
```

## 将来の拡張

### DSD over RTP
- DSD64/DSD128 のネイティブ転送
- DoP (DSD over PCM) エンコーディング

### マルチチャンネル
- 5.1ch / 7.1ch サラウンド対応
- Dolby Atmos / DTS:X（要ライセンス）

### 適応的ビットレート
- ネットワーク帯域に応じた自動ビット深度調整
- ロスレス → ロッシー自動切り替え
