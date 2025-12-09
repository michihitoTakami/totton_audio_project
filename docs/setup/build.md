# ビルド手順

Magic Box Projectの詳細なビルド手順とオプション。

## 必要なパッケージ

### Ubuntu/Debian

```bash
# 必須パッケージ
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libsndfile1-dev \
    libasound2-dev \
    libzmq3-dev

# オプション（推奨）
sudo apt install -y \
    libsystemd-dev    # systemd Type=notify サポート
```

### CUDA Toolkit

CUDAは別途インストールが必要です。

```bash
# バージョン確認
nvcc --version
nvidia-smi
```

推奨バージョン: CUDA 12.0以上

## ビルド手順

### 基本ビルド（Release）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### デバッグビルド

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

### クリーンビルド

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## CMakeオプション

### ビルドタイプ

| オプション | 説明 |
|-----------|------|
| `Release` | 最適化有効、デバッグ情報なし（本番用） |
| `Debug` | 最適化なし、デバッグ情報あり（開発用） |
| `RelWithDebInfo` | 最適化有効、デバッグ情報あり |

### CUDA アーキテクチャ

```bash
# RTX 2070 Super (Turing, SM 7.5) - デフォルト
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75

# Jetson Orin Nano (Ampere, SM 8.7)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87

# 複数アーキテクチャ
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;87"
```

### CUDAコンパイラの指定

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.0/bin/nvcc
```

## 生成されるバイナリ

### 実行ファイル

| ファイル | 説明 |
|---------|------|
| `gpu_upsampler_alsa` | メインデーモン（TCP/ALSA入力→GPU処理→ALSA出力） |
| `test_eq` | EQテストツール |

### テストバイナリ

| ファイル | 説明 | GPU必要 |
|---------|------|---------|
| `cpu_tests` | CPUユニットテスト | No |
| `gpu_tests` | GPUテスト | Yes |
| `zmq_tests` | ZeroMQ通信テスト | No |
| `auto_negotiation_tests` | Auto-Negotiationテスト | No |
| `soft_mute_tests` | Soft Muteテスト | No |
| `base64_tests` | Base64エンコードテスト | No |
| `error_codes_tests` | エラーコードテスト | No |
| `crossfeed_tests` | Crossfeedテスト | Yes |
| `fallback_manager_tests` | Fallback Managerテスト | No |

## 依存関係の自動取得

CMakeは以下のライブラリを自動的にダウンロード・ビルドします：

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| cppzmq | v4.10.0 | ZeroMQ C++バインディング |
| nlohmann_json | v3.11.3 | JSONパーサー |
| googletest | v1.14.0 | テストフレームワーク |
| spdlog | v1.14.1 | 構造化ロギング |

## オプション機能の検出

CMake実行時に以下のメッセージが表示されます：

```
-- libsystemd found - systemd notification enabled (Type=notify)
-- NVML found - GPU utilization monitoring enabled
```

### libsystemd が見つからない場合

```bash
sudo apt install libsystemd-dev
cmake -B build  # 再実行
```

systemdがなくてもビルド・実行可能ですが、`Type=notify`サポートが無効になります。

## トラブルシューティング

### CMakeがCUDAを見つけられない

```bash
# CUDAパスを明示的に指定
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cmake -B build
```

### ALSA開発ヘッダーが見つからない

```bash
sudo apt install libasound2-dev
```

### libzmq が見つからない

```bash
sudo apt install libzmq3-dev
```

### ビルド時にメモリ不足

並列ジョブ数を減らしてください：

```bash
cmake --build build -j4   # 4並列
cmake --build build -j2   # 2並列
cmake --build build       # 1並列
```

### リンクエラー（undefined reference）

依存関係が不足している可能性があります。クリーンビルドを試してください：

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Jetsonでのビルド

Jetson Orin Nano向けのビルド：

```bash
# SM 8.7 (Ampere) を指定
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build -j$(nproc)
```

詳細は [docs/jetson/README.md](../jetson/README.md) を参照。

## フィルタ係数の確認

ビルド前にフィルタ係数が生成されていることを確認：

```bash
ls -lh data/coefficients/filter_*_*x_2m_min_phase.bin
ls -lh data/coefficients/filter_*_*x_2m_linear_phase.bin

# 期待される出力（線形位相の例）:
# filter_44k_2x_2m_linear_phase.bin  (約7.6MB)
# filter_44k_4x_2m_linear_phase.bin
# filter_44k_8x_2m_linear_phase.bin
# filter_44k_16x_2m_linear_phase.bin
# filter_48k_2x_2m_linear_phase.bin
# filter_48k_4x_2m_linear_phase.bin
# filter_48k_8x_2m_linear_phase.bin
# filter_48k_16x_2m_linear_phase.bin
```

フィルタがない場合は先に生成してください：

```bash
uv run python scripts/generate_minimum_phase.py --generate-all          # 最小位相
uv run python scripts/generate_linear_phase.py --generate-all     # 線形位相（100Hz/10ms）
```
