# セットアップガイド

Magic Box Projectの開発環境構築とセットアップに関するドキュメント。

## ドキュメント一覧

| ドキュメント | 内容 | 対象者 |
|-------------|------|--------|
| [クイックスタート](quick_start.md) | 最小限の手順で動作確認 | 初めての方 |
| [ビルド手順](build.md) | 詳細なビルド手順・トラブルシューティング | 開発者 |
| [テスト実行](test.md) | テストの実行方法 | 開発者 |
| [PC開発環境](pc_development.md) | PipeWire/ALSA連携の詳細設定 | 開発者（実機動作） |
| [Web UI](web_ui.md) | Web UIサーバーの起動・設定 | 開発者 |

## 環境別ガイド

### PC開発環境（推奨）

```
┌─────────────────────────────────────────────────────────────────┐
│  PC開発環境 (Ubuntu 22.04+)                                      │
├─────────────────────────────────────────────────────────────────┤
│  GPU: RTX 2070 Super以上 (CUDA SM 7.5+)                         │
│  Audio: PipeWire → GPU Upsampler → ALSA → USB DAC               │
└─────────────────────────────────────────────────────────────────┘
```

1. [クイックスタート](quick_start.md) で環境構築
2. [ビルド手順](build.md) でビルド
3. [テスト実行](test.md) でテスト
4. [PC開発環境](pc_development.md) で実機動作確認

### Jetson Orin Nano（本番環境）

```
┌─────────────────────────────────────────────────────────────────┐
│  Jetson Orin Nano Super (Magic Box)                             │
├─────────────────────────────────────────────────────────────────┤
│  SoC: Jetson Orin Nano Super (8GB, SM 8.7)                      │
│  Audio: USB Gadget (UAC2) → GPU Upsampler → USB DAC             │
└─────────────────────────────────────────────────────────────────┘
```

→ [docs/jetson/README.md](../jetson/README.md) を参照

## 必要なソフトウェア

### 必須

| ソフトウェア | バージョン | 用途 |
|-------------|----------|------|
| CUDA Toolkit | 12.0+ | GPU計算 |
| CMake | 3.20+ | ビルドシステム |
| GCC | 11+ | C++コンパイラ |
| Python | 3.11+ | スクリプト・Web API |
| uv | 最新 | Python依存管理 |

### オプション（実機動作時）

| ソフトウェア | バージョン | 用途 |
|-------------|----------|------|
| PipeWire | 0.3+ | オーディオ入力 |
| ALSA | - | オーディオ出力 |
| ZeroMQ | 4.3+ | IPC通信 |

## セットアップの流れ

```mermaid
flowchart TD
    A[リポジトリクローン] --> B[依存関係インストール]
    B --> C[フィルタ係数生成]
    C --> D[ビルド]
    D --> E{目的は?}
    E -->|テストのみ| F[テスト実行]
    E -->|実機動作| G[PipeWire設定]
    G --> H[デーモン起動]
    H --> I[Web UI起動]
```

### Step 1: 依存関係

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake nvidia-cuda-toolkit \
    libpipewire-0.3-dev libasound2-dev libzmq3-dev

# Python (uv推奨)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Step 2: フィルタ係数生成

```bash
# 全構成（44k/48k × 2x/4x/8x/16x）一括生成
uv run python scripts/generate_filter.py --generate-all --phase-type minimum
```

### Step 3: ビルド

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Step 4: テスト

```bash
./build/cpu_tests        # CPUテスト
./build/gpu_tests        # GPUテスト
```

詳細は各ドキュメントを参照してください。
