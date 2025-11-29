# PipeWire開発環境の詳細設定

このドキュメントでは、ローカルPC上でPipeWireを使った標準的な開発環境の詳細設定について説明します。

## 概要

PipeWireモードは、PC上で直接Magic Boxデーモンを動かす最も標準的な開発環境です。

```
[Audio Source] → [PipeWire] → [Magic Box Daemon] → [USB DAC]
```

## 前提条件

- NVIDIA GPU（RTX 2070 Super以上推奨）
- CUDA 12.6以上
- Docker + NVIDIA Container Runtime
- PipeWire 1.0以上
- USB DAC

## セットアップ詳細

### 1. NVIDIA Container Runtime

```bash
# インストール確認
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi

# なければインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. PipeWireシンク作成（自動化）

手動作成の代わりに、起動スクリプトを使用:

```bash
# scripts/setup_pipewire_sink.sh を作成
#!/bin/bash
SINK_NAME="gpu_upsampler_sink"

# 既存シンク確認
if pactl list sinks short | grep -q "$SINK_NAME"; then
    echo "Sink $SINK_NAME already exists"
    exit 0
fi

# シンク作成
MODULE_ID=$(pactl load-module module-null-sink \
    sink_name=$SINK_NAME \
    sink_properties=device.description="GPU_Upsampler_Sink")

echo "Created sink $SINK_NAME (module ID: $MODULE_ID)"

# デフォルトシンクに設定
pactl set-default-sink $SINK_NAME

# 既存のストリームを移動
pactl list short sink-inputs | awk '{print $1}' | \
    xargs -I{} pactl move-sink-input {} $SINK_NAME
```

### 3. Docker Compose起動

```bash
cd docker/local/pipewire
docker compose up -d --build

# ヘルスチェック待機
docker compose ps
```

### 4. PipeWireリンク作成（自動化）

```bash
# scripts/link_pipewire.sh
#!/bin/bash

# デーモンノード出現待機
echo "Waiting for GPU Upsampler Input node..."
for i in {1..30}; do
    if pw-link -i | grep -q "GPU Upsampler Input:input_FL"; then
        echo "Node found!"
        break
    fi
    sleep 1
done

# リンク作成
pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL"
pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR"

echo "PipeWire links created successfully"
```

## 設定ファイル

### docker-compose.yml

```yaml
services:
  audio-daemon:
    image: magicbox-local:latest
    build:
      context: ../../..
      dockerfile: docker/jetson/Dockerfile.jetson
      args:
        BASE_IMAGE_DEVEL: nvidia/cuda:12.6.2-devel-ubuntu22.04
        BASE_IMAGE_RUNTIME: nvidia/cuda:12.6.2-runtime-ubuntu22.04
    container_name: magicbox-audio-local
    command: daemon
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    devices:
      - /dev/snd:/dev/snd
    shm_size: '256m'
    volumes:
      - zmq-socket:/tmp
    cap_add:
      - SYS_NICE

  web-ui:
    image: magicbox-local:latest
    container_name: magicbox-web-local
    command: web
    restart: unless-stopped
    depends_on:
      audio-daemon:
        condition: service_healthy
    ports:
      - "80:80"
    volumes:
      - zmq-socket:/tmp
    environment:
      - ZMQ_ENDPOINT=ipc:///tmp/gpu_os.sock

volumes:
  zmq-socket:
```

## トラブルシューティング

### PipeWireノードが見つからない

```bash
# すべてのノードをリスト
pw-cli list-objects

# GPU Upsampler Inputを検索
pw-cli list-objects | grep -A 10 "GPU Upsampler Input"

# デーモンログ確認
docker logs magicbox-audio-local | grep -i pipewire
```

### 音声が途切れる・ノイズが出る

```bash
# バッファサイズ確認
pw-metadata -n settings

# quantum調整（~/.config/pipewire/pipewire.conf）
default.clock.quantum = 1024
default.clock.min-quantum = 256
default.clock.max-quantum = 8192
```

### USB DACが認識されない

```bash
# ホスト側確認
aplay -l

# コンテナ内確認
docker exec magicbox-audio-local ls -la /dev/snd/

# デバイスマウント確認
docker inspect magicbox-audio-local | grep -A 10 Devices
```

### GPUメモリ不足

```bash
# GPU使用状況確認
nvidia-smi

# コンテナ内で確認
docker exec magicbox-audio-local nvidia-smi

# 他のGPUプロセス確認
nvidia-smi pmon -c 1
```

## パフォーマンスチューニング

### CPUアフィニティ設定

```yaml
# docker-compose.yml
services:
  audio-daemon:
    cpuset: "0-7"  # 8コアを割り当て
    cpu_shares: 2048
```

### メモリロック設定

```yaml
services:
  audio-daemon:
    ulimits:
      memlock:
        soft: -1
        hard: -1
```

### リアルタイム優先度

```yaml
services:
  audio-daemon:
    cap_add:
      - SYS_NICE
      - SYS_RESOURCE
```

## デバッグ

### ログレベル調整

```bash
# 環境変数でログレベル設定
docker compose down
docker compose up -d --build -e LOG_LEVEL=DEBUG

# ログ確認
docker logs -f --tail 100 magicbox-audio-local
```

### プロファイリング

```bash
# GPUプロファイリング
docker exec magicbox-audio-local nvprof ./gpu_upsampler_alsa

# CPUプロファイリング
docker exec magicbox-audio-local perf record -g ./gpu_upsampler_alsa
```

## 関連ドキュメント

- `docker/local/pipewire/README.md`: クイックスタートガイド
- `docs/setup/pc_development.md`: PC開発環境全般
- `README.md`: プロジェクト概要
