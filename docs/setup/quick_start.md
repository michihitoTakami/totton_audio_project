# クイックスタート

最小限の手順でMagic Box Projectをビルド・テストする方法。

## 前提条件

- Ubuntu 22.04以上
- NVIDIA GPU (RTX 2070 Super以上推奨)
- CUDA Toolkit 12.0以上がインストール済み

## 1. 依存関係のインストール

```bash
# システムパッケージ
sudo apt update
sudo apt install -y build-essential cmake \
    libasound2-dev libzmq3-dev

# Python環境（uv推奨）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # または新しいターミナルを開く
```

## 2. リポジトリのクローン

```bash
git clone <repository-url>
cd gpu_os
```

## 3. Python依存関係のセットアップ

```bash
uv sync
```

## 4. フィルタ係数の生成

```bash
# 全構成（44k/48k × 2x/4x/8x/16x）最小位相フィルタ
uv run python scripts/generate_minimum_phase.py --generate-all

# 全構成の線形位相フィルタ
uv run python scripts/generate_linear_phase.py --generate-all

# 生成されるファイル:
# data/coefficients/filter_44k_{2,4,8,16}x_2m_min_phase.bin
# data/coefficients/filter_44k_{2,4,8,16}x_2m_linear_phase.bin
# data/coefficients/filter_48k_{2,4,8,16}x_2m_min_phase.bin
# data/coefficients/filter_48k_{2,4,8,16}x_2m_linear_phase.bin
# 各ファイルに対応する.jsonメタデータ
```

## 5. ビルド

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 6. テスト実行

```bash
# CPUテスト（GPU不要）
./build/cpu_tests
./build/zmq_tests
./build/auto_negotiation_tests
./build/soft_mute_tests
./build/base64_tests
./build/error_codes_tests

# GPUテスト（GPU必要）
./build/gpu_tests
./build/crossfeed_tests
```

すべてのテストがパスすれば環境構築は完了です。

## 次のステップ

| 目的 | 参照ドキュメント |
|------|----------------|
| 詳細なビルドオプション | [build.md](build.md) |
| テストの詳細 | [test.md](test.md) |
| 実機で音を出す | [pc_development.md](pc_development.md) |
| Web UIを使う | [web_ui.md](web_ui.md) |

## トラブルシューティング

### CUDA関連のエラー

```bash
# CUDAバージョン確認
nvcc --version
nvidia-smi

# CUDA環境変数の設定
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CMakeがCUDAを見つけられない

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### フィルタ生成が失敗する

```bash
# メモリ不足の場合、個別に生成
uv run python scripts/generate_minimum_phase.py --input-rate 44100 --upsample-ratio 16

# 依存関係の確認
uv pip list | grep -E "numpy|scipy"
```
