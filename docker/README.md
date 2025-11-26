# Magic Box Docker環境

## 概要

このディレクトリにはJetson Orin Nano Super向けのDocker環境を含みます。

## アーキテクチャ互換性

| プラットフォーム | アーキテクチャ | 対応 |
|-----------------|---------------|------|
| Jetson Orin Nano | ARM64 (aarch64) | ✅ ネイティブビルド |
| x86_64 PC | AMD64 | ❌ クロスコンパイル要 |

### 係数binファイルの互換性

FIRフィルタ係数（`data/coefficients/*.bin`）は**両アーキテクチャで互換**です：

- 形式: IEEE 754 float32
- エンディアン: リトルエンディアン（両環境共通）
- 変換不要でそのまま使用可能

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

### x86からのクロスコンパイル（上級者向け）

NVIDIAの[JetPack Cross Compilation container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jetpack-linux-aarch64-crosscompile-x86)を使用：

```bash
# x86ホストで実行（約37.5GBのイメージをダウンロード）
docker pull nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:jp61
```

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

```bash
cd docker
docker compose up -d

# ログ確認
docker compose logs -f

# 停止
docker compose down
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
