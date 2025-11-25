# GPU Audio Upsampler セットアップガイド

## 概要

このガイドでは、GPU Audio Upsamplerを使用してPipeWireからの音声を705.6kHz (16x upsampling)でSMSL D400EX DACに出力する方法を説明します。

## システム要件

- **GPU**: NVIDIA GeForce RTX 2070 Super以上 (CUDA対応)
- **DAC**: SMSL D400EX (USB接続、705.6kHz対応)
- **OS**: Linux (PipeWire使用)
- **ソフトウェア**: CUDA Toolkit、PipeWire、ALSA

## セットアップ手順

### 1. ビルド

```bash
cd /home/michihito/Working/gpu_os
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

### 2. PipeWire Null Sinkの作成

GPU Upsamplerへ音声をルーティングするための仮想デバイスを作成します:

```bash
pw-cli create-node adapter '{
  factory.name=support.null-audio-sink
  node.name=gpu_upsampler_sink
  node.description="GPU Upsampler Sink"
  media.class=Audio/Sink
  audio.position=[FL FR]
}'
```

**重要**: この設定はPipeWire再起動時にリセットされます。永続化するには`~/.config/pipewire/pipewire.conf.d/`に設定ファイルを作成してください。

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
Creating PipeWire input (capturing from gpu_upsampler_sink)...
ALSA: Sample rate set to 705600 Hz
PipeWire input stream state: connecting
```

### 4. PipeWire Monitor接続の確立

デーモンが起動したら、PipeWireのモニター出力とGPU Upsamplerの入力を手動で接続します:

```bash
pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL"
pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR"
```

接続が成功すると、デーモンの出力が以下のように変わります:
```
PipeWire input stream state: streaming
ALSA: Output device configured (705.6kHz, 32-bit int, stereo)
```

### 5. アプリケーションの音声出力先を変更

PulseAudio/PipeWireの音量コントロール、またはGNOME設定で、再生したいアプリケーション(Spotify、Firefox等)の出力先を **"GPU Upsampler Sink"** に変更します。

```bash
# コマンドラインで確認する場合
pw-link -l | grep -A2 spotify

# 期待される出力:
# spotify:output_FL → gpu_upsampler_sink:playback_FL
# spotify:output_FR → gpu_upsampler_sink:playback_FR
```

## 動作確認

### PipeWire接続状態の確認

```bash
pw-link -l | grep -E "(gpu_upsampler|GPU)"
```

正しく設定されていれば、以下の接続が表示されます:
```
<アプリケーション>:output_FL → gpu_upsampler_sink:playback_FL
<アプリケーション>:output_FR → gpu_upsampler_sink:playback_FR
gpu_upsampler_sink:monitor_FL → GPU Upsampler Input:input_FL
gpu_upsampler_sink:monitor_FR → GPU Upsampler Input:input_FR
```

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
PipeWire input stream state: streaming
ALSA: Output device configured (705.6kHz, 32-bit int, stereo)
```

## トラブルシューティング

### 音が出ない場合

1. **PipeWire接続の確認**:
   ```bash
   pw-link -l | grep gpu_upsampler
   ```
   手順4のモニター接続が確立されているか確認してください。

2. **ALSA デバイスが使用中**:
   ```
   ALSA: Cannot open device hw:3,0: Device or resource busy
   ```
   他のプロセス(PipeWire、他のgpu_upsampler_alsa)が使用している可能性があります:
   ```bash
   # 他のデーモンプロセスを終了
   pkill gpu_upsampler_alsa

   # PipeWireがALSAデバイスを直接使用している場合は設定を確認
   pw-cli ls Node | grep -A10 "SMSL"
   ```

3. **Stream状態がPausedのまま**:
   手順4のモニター接続が欠落しています。`pw-link`コマンドで接続してください。

### クラックリングノイズ(プチプチ音)が発生する場合

以下の修正が適用されているか確認してください:

- **src/convolution_engine.cu:393**: Overlap bufferの不適切な初期化を削除
- **src/convolution_engine.cu:576-586**: CUDA streamの明示的な同期
- **src/alsa_daemon.cpp:240-244**: Float→Int32変換のオーバーフロー修正

詳細は `/home/michihito/Working/gpu_os/docs/crackling_noise_investigation.md` を参照してください。

### デーモンが起動しない

1. **CUDA環境の確認**:
   ```bash
   nvidia-smi
   ```

2. **フィルタ係数ファイルの確認**:
   ```bash
   ls -lh data/coefficients/filter_44k_16x_2m_min_phase.bin
   ```
   約7.6MBのファイルが存在する必要があります。

## 音声経路

```
[Spotify/Firefox等のアプリケーション]
        ↓ (PipeWire/PulseAudio)
[gpu_upsampler_sink] (PipeWire null sink)
        ↓ monitor output
[GPU Upsampler Input] (PipeWire stream capture)
        ↓ (リングバッファ)
[GPU Processing] (CUDA, 44.1kHz → 705.6kHz, 2M-tap FIR)
        ↓ (ALSA hw:3,0 direct)
[SMSL D400EX DAC] (705.6kHz, S32_LE)
        ↓ (アナログ出力)
[スピーカー/ヘッドフォン]
```

## 技術仕様

- **アップサンプリング比**: 16x (44.1kHz → 705.6kHz)
- **FIRフィルタ**: 2,000,000タップ minimum-phase
- **FFTサイズ**: 1,048,576サンプル
- **Overlap-Saveアルゴリズム**: オーバーラップ999,999サンプル、有効出力48,577サンプル/ブロック
- **ALSA出力フォーマット**: S32_LE (32-bit signed integer, little-endian)
- **バッファサイズ**: 16,384フレーム (ALSA)
- **Period サイズ**: 4,096フレーム (ALSA)

## 永続化設定(オプション)

### PipeWire Null Sinkの自動作成

`~/.config/pipewire/pipewire.conf.d/99-gpu-upsampler-sink.conf`:

```
context.objects = [
    {
        factory = adapter
        args = {
            factory.name           = support.null-audio-sink
            node.name              = "gpu_upsampler_sink"
            node.description       = "GPU Upsampler Sink"
            media.class            = "Audio/Sink"
            audio.position         = [ FL FR ]
        }
    }
]
```

### Systemdサービス化

`~/.config/systemd/user/gpu-upsampler.service`:

```ini
[Unit]
Description=GPU Audio Upsampler Daemon
After=pipewire.service

[Service]
Type=simple
WorkingDirectory=/home/michihito/Working/gpu_os
ExecStart=/home/michihito/Working/gpu_os/build/gpu_upsampler_alsa
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

**注意**: Monitor接続(`pw-link`コマンド)は現在手動で行う必要があります。自動化スクリプトの作成を推奨します。

## 関連ドキュメント

- [クラックリングノイズ調査報告](crackling_noise_investigation.md)
- [Phase 1 実装報告](phase1_implementation_report.md)

## 作成日時

2025-11-21
