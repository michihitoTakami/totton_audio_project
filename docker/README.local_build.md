# Docker（ローカルビルド / 開発者向け）

このREADMEは **開発者向け**です。ソースコードを用いてローカル（Jetson上）でビルドして起動します。

> 評価者（ソース不要）で起動したい場合は [README.evaluator.md](./README.evaluator.md) を参照してください。

---

## Jetson Compose（ローカルビルド）

Jetson 上で Totton Audio Project を起動する構成です。

- **I2Sメイン運用**のため、RTP はデフォルトで無効です（`TOTTON_AUDIO_ENABLE_RTP=false`）。
- RTP を使う場合（フォールバック等）は `TOTTON_AUDIO_ENABLE_RTP=true` にし、必要なら `TOTTON_AUDIO_RTP_AUTOSTART=true` で自動起動します。

```bash
cd docker

# 既定: secure-by-default（localhost bind）
docker compose -f jetson/docker-compose.jetson.yml up -d
docker compose -f jetson/docker-compose.jetson.yml logs -f

# 停止
docker compose -f jetson/docker-compose.jetson.yml down
```

---

## Totton Audio Project コンテナの事前ビルド（Jetson 本体で実行）

`docker/jetson/Dockerfile.jetson` はホストでビルド済みのバイナリをコピーする前提です。
コンテナを立ち上げる前に Jetson 上で以下を実行し、`build/gpu_upsampler_alsa` を用意してください。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
ls -l build/gpu_upsampler_alsa
```

ビルド手順の詳細は `../docs/setup/build.md` を参照してください。
