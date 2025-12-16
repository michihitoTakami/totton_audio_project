# GPU Audio Upsampler セットアップガイド

## 概要

このガイドでは、GStreamer RTP で Jetson に入力を届け、ALSA Loopback 経由で GPU パイプラインへ流す方法を説明します。PC 単体での開発時は、従来通り ALSA Loopback を使ったローカル入力も利用できます。システムは入力レートと DAC 性能に基づいて自動的に最適なアップサンプリング倍率（2x/4x/8x/16x）を選択します。

## システム要件

- **GPU**: NVIDIA GeForce RTX 2070 Super以上 (CUDA対応)
- **DAC**: SMSL D400EX (USB接続、705.6kHz対応)
- **OS**: Linux (ALSA)
- **ソフトウェア**: CUDA Toolkit、ALSA、(任意) ZeroMQ

## セットアップ手順

### 1. ビルド

```bash
# プロジェクトルートで実行
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2. ALSA Loopbackの有効化（任意）

PC上で簡単に入力を与えるには ALSA Loopback を使います。`snd-aloop` がロードされていない場合は以下を実行してください。

```bash
sudo modprobe snd-aloop
```

`/proc/asound/cards` に `Loopback` が見えればOKです。

### 3. GPU Upsampler Daemonの起動

```bash
./build/gpu_upsampler_alsa
```

起動時の出力例:
```
========================================
  GPU Audio Upsampler - ALSA Direct Output
  44.1kHz → 705.6kHz (16x upsampling)
========================================

Initializing GPU upsampler...
GPU upsampler ready (16x upsampling, 4096 samples/block)

Starting ALSA output thread...
```

### 4. 音声を入力する方法

#### (推奨) ALSA Loopback を使う場合
1. ループバックの playback 側に音源を送る
   ```bash
   # 例: 48kHz ステレオ WAV を送る
   aplay -D hw:Loopback,0,0 test.wav
   ```
2. `config.json` の `loopback.enabled` を `true` にすると、デーモンが capture 側 (`hw:Loopback,1,0`) から読み込みます。

#### (Jetsonでのネットワーク入力) RTP を使う場合
Jetson では `rtp_input` サービス（GStreamer）で RTP を受信し、ALSA Loopback playback に書き込みます。Magic Box Web API から起動できます。

```bash
# Jetson 上で RTP 受信を開始
curl -X POST http://localhost:8000/api/rtp-input/start

# 設定変更の例
curl -X PUT http://localhost:8000/api/rtp-input/config \
  -H 'Content-Type: application/json' \
  -d '{"sample_rate":44100,"encoding":"L24","latency_ms":100}'
```

送信側 (Raspberry Pi) は `raspberry_pi/rtp_sender.py` または `docker-compose.yml` の `rtp-sender` サービスを使用してください。

## 動作確認

### ALSA出力デバイスの確認

```bash
cat /proc/asound/cards
```

SMSL DACが認識されていることを確認します:
```
 3 [AUDIO          ]: USB-Audio - SMSL USB AUDIO
                      SMSL SMSL USB AUDIO at usb-0000:00:14.0-3.3, high speed
```

デーモンは`hw:3,0`に直接出力します。

### GPU処理の確認

デーモンログで以下を確認:
```
GPU upsampler ready (16x upsampling, 4096 samples/block)
ALSA: Output device configured (705.6kHz, 32-bit int, stereo)
```

## トラブルシューティング

### 音が出ない場合

1. **Loopback/入力経路の確認**:
   ```bash
   aplay -D hw:Loopback,0,0 /dev/zero 2>/dev/null | head
   ```
   ループバック playback に送った音が capture 側で読めるか確認してください。

2. **ALSA デバイスが使用中**:
   ```
   ALSA: Cannot open device hw:3,0: Device or resource busy
   ```
   他のプロセスがデバイスを使用していないか確認してください:
   ```bash
   # 他のデーモンプロセスを終了
   pkill gpu_upsampler_alsa

   # どのプロセスがデバイスを掴んでいるか確認
   lsof /dev/snd/pcmC3D0p
   ```

3. **ストリームが途切れる**:
   送信側の RTP/Loopback が止まっていないか確認してください。必要に応じて再送出してください。

### クラックリングノイズ(プチプチ音)が発生する場合

以下の修正が適用されているか確認してください:

- **src/convolution_engine.cu:393**: Overlap bufferの不適切な初期化を削除
- **src/convolution_engine.cu:576-586**: CUDA streamの明示的な同期
- **src/entrypoints/alsa_daemon.cpp:240-244**: Float→Int32変換のオーバーフロー修正

詳細は [クラックリングノイズ調査報告](../investigations/crackling_noise_investigation.md) を参照してください。

### デーモンが起動しない

1. **CUDA環境の確認**:
   ```bash
   nvidia-smi
   ```

2. **フィルタ係数ファイルの確認**:
   ```bash
   # マルチレート対応フィルタの確認
   ls -lh data/coefficients/filter_*_*x_2m_linear_phase.bin
   ```
   各フィルタファイル（約7.6MB）と対応するメタデータJSONファイルが存在する必要があります。

   ```bash
   # メタデータの確認（DCゲイン・レート情報）
   cat data/coefficients/filter_44k_16x_2m_linear_phase.json | jq '.sample_rate_input, .sample_rate_output, .upsample_ratio, .validation_results.normalization.normalized_dc_gain'
   ```

## 音声経路

```
[音源] ──> ALSA Loopback playback (hw:Loopback,0,0)
        └─> ALSA Loopback capture (hw:Loopback,1,0) → GPU Upsampler (16x) → ALSA → DAC
```

## 技術仕様

### マルチレート対応

システムは入力レートとDAC性能に基づいて自動的に最適なアップサンプリング倍率を選択します：

| 入力レート | 出力レート | 倍率 | フィルタファイル |
|-----------|----------|------|----------------|
| 44.1kHz | 705.6kHz | 16x | filter_44k_16x_2m_linear_phase.bin |
| 88.2kHz | 705.6kHz | 8x | filter_44k_8x_2m_linear_phase.bin |
| 176.4kHz | 705.6kHz | 4x | filter_44k_4x_2m_linear_phase.bin |
| 352.8kHz | 705.6kHz | 2x | filter_44k_2x_2m_linear_phase.bin |
| 48kHz | 768kHz | 16x | filter_48k_16x_2m_linear_phase.bin |
| 96kHz | 768kHz | 8x | filter_48k_8x_2m_linear_phase.bin |
| 192kHz | 768kHz | 4x | filter_48k_4x_2m_linear_phase.bin |
| 384kHz | 768kHz | 2x | filter_48k_2x_2m_linear_phase.bin |

### フィルタとゲイン設定

- **FIRフィルタ**: 2,000,000タップ minimum-phase（各レート・倍率に対応）
- **DCゲイン**: 各フィルタはアップサンプリング倍率 × 0.99に正規化されており、全レートで音量が統一されています
- **config.json**: デフォルトゲインは`1.0`に設定されており、全レートで適切に動作します
- **メタデータ**: 各フィルタには対応する`.json`ファイルがあり、DCゲイン、入力/出力レート、アップサンプリング倍率が記載されています

### 処理パラメータ

- **FFTサイズ**: 1,048,576サンプル
- **Overlap-Saveアルゴリズム**: オーバーラップ999,999サンプル、有効出力48,577サンプル/ブロック
- **ALSA出力フォーマット**: S32_LE (32-bit signed integer, little-endian)
- **バッファサイズ**: 16,384フレーム (ALSA)
- **Period サイズ**: 4,096フレーム (ALSA)

## 永続化設定(オプション)

### Systemdサービス化

`~/.config/systemd/user/gpu-upsampler.service`:

```ini
[Unit]
Description=GPU Audio Upsampler Daemon
After=sound.target

[Service]
Type=simple
WorkingDirectory=/path/to/gpu_os
ExecStart=/path/to/gpu_os/build/gpu_upsampler_alsa
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

有効化:
```bash
systemctl --user daemon-reload
systemctl --user enable --now gpu-upsampler.service
```

**注意**: ALSA デバイス占有で失敗する場合は、他プロセスがデバイスを掴んでいないか確認してください。

## 関連ドキュメント

- [クラックリングノイズ調査報告](../investigations/crackling_noise_investigation.md)
- [ビルド手順](build.md)
- [Web UI セットアップ](web_ui.md)
