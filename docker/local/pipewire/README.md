# PipeWire開発環境（ローカル）

このディレクトリには、PC上でPipeWire入力から直接Magic Boxデーモンを動かす標準的な開発環境が含まれています。

## 概要

```
[Audio Source (Spotify, Chrome, etc)]
    ↓
[PipeWire]
    ↓
[Magic Box Daemon (Docker)]
 - GPU Convolution
 - EQ Processing
    ↓
[USB DAC]
```

## 前提条件

- NVIDIA GPU（CUDA対応）
- Docker + NVIDIA Container Runtime
- PipeWire実行中
- USB DAC接続済み

## セットアップ

### 1. PipeWireシンクの作成

Magic Boxデーモンが読み取るPipeWireシンクを作成します：

```bash
# gpu_upsampler_sink を作成
pactl load-module module-null-sink \
    sink_name=gpu_upsampler_sink \
    sink_properties=device.description="GPU_Upsampler_Sink"
```

### 2. Docker起動

```bash
cd docker/local/pipewire
docker compose up -d --build
```

### 3. PipeWireリンク作成

```bash
# MonitorをMagic Boxデーモンに接続
pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL"
pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR"
```

### 4. デフォルトシンクに設定

```bash
pactl set-default-sink gpu_upsampler_sink
```

## 動作確認

### Magic Boxダッシュボードにアクセス

```bash
# ブラウザで開く
xdg-open http://localhost:80
```

### ログ確認

```bash
# デーモンログ
docker logs -f magicbox-audio-local

# Web UIログ
docker logs -f magicbox-web-local
```

## トラブルシューティング

### GPU が認識されない

```bash
# NVIDIA Container Runtime確認
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

### PipeWireリンクが作成できない

```bash
# ノードを確認
pw-cli list-objects | grep -A 5 "GPU Upsampler"
pw-cli list-objects | grep -A 5 "gpu_upsampler_sink"

# リンク状態確認
pw-link -l
```

### USB DACが認識されない

```bash
# ALSAデバイス確認
aplay -l

# Dockerコンテナ内で確認
docker exec magicbox-audio-local aplay -l
```

## 関連ドキュメント

- `docs/setup/pc_development.md`: PC開発環境の詳細
- `docs/setup/local/pipewire_development.md`: PipeWireモードの詳細設定
