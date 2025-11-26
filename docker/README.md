# Magic Box Docker環境

## 概要

このディレクトリにはJetson Orin Nano Super向けのDocker環境を含みます。

## アーキテクチャ互換性

| プラットフォーム | アーキテクチャ | 対応 |
|-----------------|---------------|------|
| Jetson Orin Nano | ARM64 (aarch64) | ✅ デフォルト |
| x86_64 PC | AMD64 | ✅ `--build-arg`でx86 CUDAベースを指定 |

### 係数binファイルの互換性

FIRフィルタ係数（`data/coefficients/*.bin`）は**両アーキテクチャで互換**です：

- 形式: IEEE 754 float32
- エンディアン: リトルエンディアン（両環境共通）
- 変換不要でそのまま使用可能

EQプロファイル（`data/EQ/*.txt`）もイメージ内へ同梱しているため、`config.json`で相対パス指定するだけで使用できます。

## 前提条件

### Jetson側

- JetPack 6.1以上
- Docker + NVIDIA Container Toolkit
- 十分なストレージ（イメージサイズ: 約5GB）

```bash
# NVIDIA Container Toolkitの確認
docker run --rm --runtime=nvidia nvidia-smi
```

## ビルド

### Jetson上でのビルド（推奨）

```bash
# プロジェクトルートで実行
docker build -f docker/Dockerfile.jetson -t magicbox:latest .
```

ビルド時間目安: 約15-30分（初回）

### x86ホストでのビルド

Jetson用JetPackイメージはARM64専用のため、x86では`nvidia/cuda`ベースに切り替えてビルドします。

```bash
docker build \
  --build-arg BASE_IMAGE_DEVEL=nvidia/cuda:12.6.2-devel-ubuntu22.04 \
  --build-arg BASE_IMAGE_RUNTIME=nvidia/cuda:12.6.2-runtime-ubuntu22.04 \
  -f docker/Dockerfile.jetson \
  -t magicbox:x86 .
```

> NOTE: 生成されるバイナリはx86向けですが、FIR係数や設定ファイルはARM64と共通です。

## 実行

### 単体実行

```bash
# Web UIのみ起動
docker run --runtime=nvidia -p 80:80 magicbox:latest web

# Audio Daemonのみ起動（オーディオデバイス必要）
docker run --runtime=nvidia --device /dev/snd magicbox:latest daemon

# 両方起動
docker run --runtime=nvidia --device /dev/snd -p 80:80 magicbox:latest all

# インタラクティブシェル
docker run --runtime=nvidia -it magicbox:latest bash
```

### Docker Compose（本番環境推奨）

Jetson向けとx86ローカル向けでComposeファイルを分けています。間違ったベースイメージでビルドする事故を防ぐため、目的に合わせて`-f`で明示的に選択してください。

#### Jetson Orin (ARM64)

```bash
cd docker
docker compose -f docker-compose.jetson.yml up -d
docker compose -f docker-compose.jetson.yml logs -f
docker compose -f docker-compose.jetson.yml down
```

#### x86_64ローカル検証

```bash
cd docker
docker compose -f docker-compose.local.yml up -d --build
docker compose -f docker-compose.local.yml logs -f
docker compose -f docker-compose.local.yml down
```

## ポート一覧

| ポート | 用途 |
|--------|------|
| 80 | Web UI (HTTP) |

## ボリューム

| パス | 用途 |
|------|------|
| `/tmp/gpu_os.sock` | ZeroMQ IPC ソケット |
| `/opt/magicbox/data/coefficients` | FIRフィルタ係数 |
| `/opt/magicbox/config.json` | 設定ファイル |

## トラブルシューティング

### GPUが認識されない

```bash
# NVIDIAランタイムの確認
docker run --rm --runtime=nvidia nvidia-smi

# 出力がなければNVIDIA Container Toolkitを再インストール
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### オーディオデバイスにアクセスできない

```bash
# ホストのオーディオデバイス確認
ls -la /dev/snd/

# コンテナ内での確認
docker run --device /dev/snd -it magicbox:latest aplay -l
```

### ビルドが失敗する

```bash
# キャッシュをクリアして再ビルド
docker build --no-cache -f docker/Dockerfile.jetson -t magicbox:latest .
```

## 開発ワークフロー

推奨開発フロー：

```
[開発PC (x86_64)]           [Jetson Orin (ARM64)]
       │                            │
   コード編集                         │
   Python係数生成                     │
       │                            │
       ├──── git push ──────────────>│
       │                            ▼
       │                    git pull
       │                    docker build
       │                    docker compose up
       │                            │
       │<──── 動作確認 ──────────────┤
```

## CUDAアーキテクチャ

| デバイス | SM | CUDA_ARCHITECTURES |
|---------|----|--------------------|
| Jetson Orin Nano | 8.7 | 87 |
| RTX 2070 Super | 7.5 | 75 |
| RTX 3080 | 8.6 | 86 |
| RTX 4090 | 8.9 | 89 |

CMakeでアーキテクチャを指定：

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87
```

## 関連ドキュメント

- [../docs/jetson/README.md](../docs/jetson/README.md) - Jetson組み込みガイド
- [../docs/jetson/deployment/ota-update.md](../docs/jetson/deployment/ota-update.md) - OTAアップデート
- [../CLAUDE.md](../CLAUDE.md) - 開発ガイドライン
