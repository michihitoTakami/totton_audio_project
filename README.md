# Totton Audio Project

Delivering personalized 1-to-1 audio experiences with a GPU-powered upsampler.
1to1オーディオで一人ひとりに最適化された音楽体験を、箱をつなぎ管理画面で設定するだけで実現します。

## Overview / 概要
- GPU-accelerated upsampling with **640k-tap minimum-phase FIR** (Kaiser β≈28, ~160dB stopband), output up to 768kHz.
  GPUで640kタップ最小位相FIRを実行し、最大768kHzまでアップサンプリング。
- **Headphone EQ** using OPRA data + KB5000_7 target; **crossfeed / AI de-limiter** and safe auto-negotiation & soft mute.
  OPRA由来EQとターゲットカーブによるヘッドホン補正、クロスフィード・AIデリミッタ、レート自動切替とソフトミュート。
- Runs as a standalone DDC/DSP on Jetson Orin Nano or on PC for development.
  Jetson Orin Nanoを本番、PCを開発用として単体DDC/DSPとして動作。

## Quick Start (Docker-first) / クイックスタート（Docker推奨）
Dockerを前提に最短で動かす手順をまとめています。詳細・派生構成は各ドキュメントを参照してください。

### Jetson runtime (prebuilt images) / Jetson実機（プリビルト）
- Start the GPU upsampler on Jetson with the provided Compose file.
  Jetson上でGPUアップサンプラーをDocker Composeで起動します。
```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml up -d
docker compose -f jetson/docker-compose.jetson.yml logs -f  # follow logs
# 停止: docker compose -f jetson/docker-compose.jetson.yml down
```

### Raspberry Pi as UAC2 bridge / Raspberry PiをUAC2受け口に
- Use the runtime-only images (no source build) to bridge USB (UAC2) input to I2S or RTP fallback.
  ソース不要のランタイムイメージで、USB(UAC2)入力をI2Sへ橋渡し（RTPはフォールバック）。
- If you connect a PC directly to the Pi over USB, the Pi must be prepared as a **UAC2 gadget** in advance.
  PCとPiをUSB直結する場合、Pi側は **UAC2 gadget** として事前設定が必要です（現状は手動セットアップ）。
```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
# 停止: docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml down
```
- For local builds on Pi (developers):
  開発者がPiでローカルビルドする場合:
```bash
docker compose -f raspberry_pi/docker-compose.yml up -d --build
```

### PC / developer build (optional) / PC開発者向け（任意）
- Generate filters and build locally when you need code changes or tests.
  コード改修やテストが必要な場合のみローカルビルドを行います。
```bash
uv sync
uv run python scripts/filters/generate_minimum_phase.py --generate-all
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Raspberry Pi as UAC2 input: setup essentials / ラズパイをUAC2受け口にする要点
1) Identify the UAC2 ALSA device
　`arecord -l` や `cat /proc/asound/cards` で UAC2 デバイスを確認し、`USB_I2S_CAPTURE_DEVICE` に設定します（例: `hw:CARD=UAC2Gadget,DEV=0`）。
2) Recommended format / 推奨設定
　2ch、44.1k/48k 系のレート追従、量子化は **S32_LE 推奨**。入力フォーマットは変換せず下流へ橋渡しします。
3) Persistent config / 設定ファイル
　設定は `/var/lib/usb-i2s-bridge/config.env` に集約され、初回は `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` からシードされます。
4) UAC2 gadget mode (required for PC→Pi USB) / UAC2ガジェットモード（PC→Pi USB直結時は必須）
　PCからPiへUSB一発接続で音を入れる場合は、ConfigFS等でUAC2ガジェットを構成し、上記のALSAデバイスとして見えるようにしてください（現状この自動化は未整備）。
5) Network defaults / ネットワーク既定値
　Jetson Web: `http://192.168.55.1/`、Pi Control API: `http://192.168.55.100:8081`（必要に応じ `TOTTON_AUDIO_PI_API_BASE` / `RPI_CONTROL_BIND_HOST` を上書き）。

詳しくは `docs/setup/pi_bridge.md` と `raspberry_pi/README.md` を参照してください。

## Architecture snapshot / アーキテクチャ概要
```
Control Plane (Python/FastAPI)
├── FastAPI Web API / Web UI
├── Filter Generator (scipy)
└── ZeroMQ Command Interface
   ↕
Data Plane (C++/CUDA)
├── ALSA / RTP input
├── Rate Detection → Filter Selection
├── GPU FFT Convolution (640k tap + Overlap-Save)
├── Crossfeed + Soft Mute + Headroom Limiter
└── ALSA output
```

## Documentation / ドキュメント
- Setup / セットアップ: `docs/setup/README.md`, `docs/setup/build.md`, `docs/setup/pi_bridge.md`, `docs/setup/delimiter.md`
- Deployment (Docker) / デプロイ（Docker）: `docker/README.md`, `docker/README.evaluator.md`, `docker/README.local_build.md`, `raspberry_pi/README.md`
- API: `docs/api/README.md`, `docs/api/openapi.json`, `docs/api/raspi_openapi.json`

## License & third-party data / ライセンス

| アイテム | ライセンス | 備考 |
|---------|------------|------|
| 本プロジェクトコード | MB-NCL v1.0 | 非商用・組織評価/PoC |
| OPRA EQ データ | CC BY-SA 4.0 | 帰属表示必須 |
| HUTUBS (Crossfeed) | CC BY 4.0 | 帰属表示必須 |

依存ライブラリ: CUDA/cuFFT (NVIDIA EULA), alsa-lib (LGPL-2.1), ZeroMQ (LGPL-3.0), FastAPI (MIT), scipy (BSD-3-Clause) など。EQデータは **OPRA由来のみ**を同梱します。
