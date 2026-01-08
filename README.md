# Totton Audio Project

## English

**Vision**: Delivering personalized 1-to-1 audio experiences with a GPU-powered upsampler.

### Overview

- GPU-accelerated upsampling with **640k-tap minimum-phase FIR** (Kaiser β≈28, ~160dB stopband), output up to 768kHz.
- Headphone EQ using OPRA data + KB5000_7 target; crossfeed / AI de-limiter; safe auto-negotiation & soft mute.
- Runs as a standalone DDC/DSP on Jetson Orin Nano or on PC for development.

### Quick Start (Docker-first)

#### Jetson runtime (prebuilt images)

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml up -d
docker compose -f jetson/docker-compose.jetson.yml logs -f
# stop: docker compose -f jetson/docker-compose.jetson.yml down
```

#### Raspberry Pi as UAC2 bridge

- If you connect a PC directly to the Pi over USB, the Pi must be prepared as a **UAC2 gadget** in advance (currently manual setup).

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
# stop: docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml down
```

For local builds on Pi (developers):

```bash
docker compose -f raspberry_pi/docker-compose.yml up -d --build
```

#### PC / developer build (optional)

```bash
uv sync
uv run python scripts/filters/generate_minimum_phase.py --generate-all
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Raspberry Pi as UAC2 input: setup essentials

1) Identify the UAC2 ALSA device and set `USB_I2S_CAPTURE_DEVICE` (e.g. `hw:CARD=UAC2Gadget,DEV=0`).
2) Recommended format: 2ch, 44.1k/48k families, prefer `S32_LE`, and bridge without format conversion.
3) Persistent config: `/var/lib/usb-i2s-bridge/config.env` (seeded from `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` on first run).
4) UAC2 gadget mode (required for PC→Pi USB): configure via ConfigFS so it appears as an ALSA device.
5) Network defaults: Jetson Web `http://192.168.55.1/`, Pi Control API `http://192.168.55.100:8081`.

See `docs/setup/pi_bridge.md` and `raspberry_pi/README.md`.

### Architecture snapshot

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

### Documentation

- Setup: `docs/setup/build.md`, `docs/setup/pi_bridge.md`, `docs/setup/delimiter.md`
- Deployment (Docker): `docker/README.md`, `docker/README.evaluator.md`, `docker/README.local_build.md`, `raspberry_pi/README.md`
- API: `docs/api/README.md`, `docs/api/openapi.json`, `docs/api/raspi_openapi.json`

### License & third-party data

| Item | License | Notes |
|------|---------|-------|
| Project code | MB-NCL v1.0 | Non-commercial / org evaluation / PoC |
| OPRA EQ data | CC BY-SA 4.0 | Attribution required |
| HUTUBS (Crossfeed) | CC BY 4.0 | Attribution required |

Dependencies include CUDA/cuFFT (NVIDIA EULA), alsa-lib (LGPL-2.1), ZeroMQ (LGPL-3.0), FastAPI (MIT), scipy (BSD-3-Clause). Bundled EQ data is **OPRA-only**.

---

## 日本語

**ビジョン**: 1to1オーディオで一人ひとりに最適化された音楽体験を、GPUアップサンプラーで届ける。

### 概要

- **640kタップ最小位相FIR**（Kaiser β≈28、阻止域 ~160dB）によるGPUアップサンプリング（最大768kHz出力）
- OPRA由来EQ + KB5000_7ターゲットによるヘッドホン補正、クロスフィード / AIデリミッタ、レート自動切替とソフトミュート
- Jetson Orin Nanoを本番、PCを開発用として単体DDC/DSPとして動作

### クイックスタート（Docker推奨）

#### Jetson実機（プリビルト）

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml up -d
docker compose -f jetson/docker-compose.jetson.yml logs -f
# 停止: docker compose -f jetson/docker-compose.jetson.yml down
```

#### Raspberry PiをUAC2受け口に（ブリッジ）

- PC→Pi をUSB直結で入力する場合、Pi側は **UAC2 gadget の事前設定が必須**です（現状は手動セットアップ）。

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
# 停止: docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml down
```

Piでローカルビルドする場合（開発者向け）:

```bash
docker compose -f raspberry_pi/docker-compose.yml up -d --build
```

#### PCでソースからビルド（任意）

```bash
uv sync
uv run python scripts/filters/generate_minimum_phase.py --generate-all
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Raspberry PiをUAC2受け口にする要点

1) `arecord -l` / `cat /proc/asound/cards` でUAC2デバイスを特定し、`USB_I2S_CAPTURE_DEVICE` を設定（例: `hw:CARD=UAC2Gadget,DEV=0`）
2) 推奨: 2ch、44.1k/48k系に追従、量子化は `S32_LE` 推奨、フォーマット変換せずに橋渡し
3) 設定ファイルは `/var/lib/usb-i2s-bridge/config.env`（初回は `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` からシード）
4) PC→Pi USB直結時（必須）: ConfigFS等でUAC2 gadgetを構成し、ALSAデバイスとして見えるようにする
5) ネットワーク既定値: Jetson Web `http://192.168.55.1/`、Pi Control API `http://192.168.55.100:8081`

詳細: `docs/setup/pi_bridge.md` と `raspberry_pi/README.md`

### アーキテクチャ概要

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

### ドキュメント

- セットアップ: `docs/setup/build.md`, `docs/setup/pi_bridge.md`, `docs/setup/delimiter.md`
- デプロイ（Docker）: `docker/README.md`, `docker/README.evaluator.md`, `docker/README.local_build.md`, `raspberry_pi/README.md`
- API: `docs/api/README.md`, `docs/api/openapi.json`, `docs/api/raspi_openapi.json`

### ライセンスとサードパーティデータ

| アイテム | ライセンス | 備考 |
|---------|------------|------|
| 本プロジェクトコード | MB-NCL v1.0 | 非商用・組織評価/PoC |
| OPRA EQ データ | CC BY-SA 4.0 | 帰属表示必須 |
| HUTUBS (Crossfeed) | CC BY 4.0 | 帰属表示必須 |

依存ライブラリ: CUDA/cuFFT (NVIDIA EULA), alsa-lib (LGPL-2.1), ZeroMQ (LGPL-3.0), FastAPI (MIT), scipy (BSD-3-Clause) など。EQデータは **OPRA由来のみ**を同梱します。
