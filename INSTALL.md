# Installation Guide

GPU Audio Upsamplerのセットアップ手順です。

## Prerequisites

- Linux (Ubuntu 22.04+ 推奨)
- NVIDIA GPU (Compute Capability 7.5+, RTX 2070以上)
- NVIDIA Driver (535+)

## Quick Start

```bash
# 1. システム依存関係のインストール (sudo必要)
./scripts/deployment/setup-system.sh

# 2. aquaのインストール (CLIツール管理)
curl -sSfL https://raw.githubusercontent.com/aquaproj/aqua-installer/v3.0.1/aqua-installer | bash
export PATH="${AQUA_ROOT_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/aquaproj-aqua}/bin:$PATH"

# 3. CLIツールのインストール (cmake, uv, gh)
aqua i

# 4. Python環境のセットアップ
uv sync

# 5. フィルタ係数の生成
uv run python scripts/filters/generate_minimum_phase.py --taps 1000000

# 6. ビルド
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 7. テスト実行
./build/cpu_tests
./build/gpu_tests
```

## Step-by-Step Installation

### 1. システム依存関係

以下のパッケージはaquaでは管理できないため、aptでインストールします：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    nvidia-cuda-toolkit \
    libsndfile1-dev \
    libasound2-dev \
    git
```

または、セットアップスクリプトを使用：

```bash
./scripts/deployment/setup-system.sh
```

> **Note**: このスクリプトはUbuntu/Debian、Fedora、Arch Linuxに対応しています。

### 2. aqua (CLIツールマネージャー)

[aqua](https://aquaproj.github.io/)を使用して、cmake, uv, ghのバージョンを固定します。

```bash
# aquaのインストール
curl -sSfL https://raw.githubusercontent.com/aquaproj/aqua-installer/v3.0.1/aqua-installer | bash

# PATHに追加 (~/.bashrc または ~/.zshrc に追記)
export PATH="${AQUA_ROOT_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/aquaproj-aqua}/bin:$PATH"

# シェルを再起動、または
source ~/.bashrc  # or source ~/.zshrc
```

### 3. CLIツールのインストール

```bash
# プロジェクトディレクトリで実行
cd /path/to/totton_audio
aqua i
```

これにより以下がインストールされます：
- **cmake** v3.30.0 - ビルドシステム
- **uv** v0.5.4 - Python環境管理
- **gh** v2.62.0 - GitHub CLI

### 4. Python環境

```bash
# 依存関係のインストール
uv sync

# 開発用依存関係も含める場合
uv sync --all-groups
```

### 5. フィルタ係数の生成

```bash
# 1M-tap minimum phase FIRフィルタを生成
uv run python scripts/filters/generate_minimum_phase.py --taps 1000000

# 2M-tapを生成する場合 (メモリ8GB以上推奨)
uv run python scripts/filters/generate_minimum_phase.py --taps 2000000 --kaiser-beta 55
```

生成されるファイル：
- `data/coefficients/filter_*_linear_phase.bin` - バイナリ係数
- `data/coefficients/filter_*_linear_phase.json` - メタデータ
- `plots/analysis/*.png` - 検証用プロット

### 6. ビルド

```bash
# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Debug build (開発時)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

### 7. 動作確認

```bash
# CPU テスト (GPU不要)
./build/cpu_tests

# GPU テスト (NVIDIA GPU必要)
./build/gpu_tests

# Python テスト
uv run pytest
```

## Managed Tools Summary

| Tool | Managed by | Version |
|------|------------|---------|
| cmake | aqua | 3.30.0 |
| uv | aqua | 0.5.4 |
| gh | aqua | 2.62.0 |
| Python packages | uv | pyproject.toml |
| CUDA toolkit | apt | system |
| libsndfile | apt | system |
| ALSA | apt | system |

## Troubleshooting

### aquaコマンドが見つからない

```bash
# PATHを確認
echo $PATH | tr ':' '\n' | grep aqua

# なければ追加
export PATH="${AQUA_ROOT_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/aquaproj-aqua}/bin:$PATH"
```

### CUDA toolkit not found

```bash
# CUDAがインストールされているか確認
nvcc --version
nvidia-smi

# なければインストール
sudo apt install nvidia-cuda-toolkit
```


```bash
# pkg-configで確認

# なければインストール
```

## Next Steps

セットアップ完了後は以下を参照：
- [Docker](docker/README.md) - まず動かす（評価者/開発者）
- [セットアップガイド](docs/setup/README.md) - セットアップ導線（Docker推奨）
- [Raspberry Pi ブリッジ](docs/setup/pi_bridge.md) - PiをUAC2受け口として初期化

## Files Added by This Setup

```
totton_audio/
├── aqua.yaml              # CLIツールバージョン定義
├── scripts/
│   └── setup-system.sh    # システム依存関係インストール
└── INSTALL.md             # このファイル
```
