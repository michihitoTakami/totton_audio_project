# GHCR 公開確認 / タグ・ダイジェスト運用

Issue: #1194（Epic: #1051）

## 目的

- 評価者が **認証なし**で GHCR からイメージを pull できることを確認する
- `linux/arm64`（Jetson）と `linux/amd64`（開発PC/NVIDIA GPU, 発行済みの場合）で **同じタグ**を使えるようにし、リリースごとにダイジェストで再現性を担保する
- Compose は `image:` 前提とし、Release アセットに含めるダイジェストで pin できるようにする

## 対象パッケージ（GHCR）

- `ghcr.io/michihitotakami/totton-audio-system`
  - `magicbox` runtime（Jetson/PC 共通タグ、`linux/arm64`。`linux/amd64` は発行されたリリースで同タグを共有予定）
  - Raspberry Pi 付帯サービス（同パッケージ内のタグで分岐）
    - `raspi-usb-i2s-bridge-*`（`linux/arm64`）
    - `raspi-control-api-*`（`linux/arm64`）

## 公開（認証なし）確認手順

以下はいずれも **docker login なし**で成功することを確認する:

```bash
# Jetson/arm64 イメージの pull（digest pin）
docker pull --platform=linux/arm64 ghcr.io/michihitotakami/totton-audio-system@sha256:<jetson-digest>

# PC/amd64 イメージの pull（digest pin、amd64 イメージを発行しているリリースのみ）
docker pull --platform=linux/amd64 ghcr.io/michihitotakami/totton-audio-system@sha256:<amd64-digest>

# Raspberry Pi 付帯サービス（arm64）
docker pull ghcr.io/michihitotakami/totton-audio-system@sha256:<raspi-bridge-digest>
docker pull ghcr.io/michihitotakami/totton-audio-system@sha256:<raspi-control-digest>
```

> いずれかが 401/404 になる場合、GHCR パッケージの可視性を `public` に設定する（`gh api --method PUT /user/packages/container/totton-audio-system/visibility -f visibility=public`）。

## タグ / ダイジェストの運用ルール

- タグ（共通）
  - **推奨**: `vX.Y.Z`（リリースタグ） / `X.Y.Z`
  - **スモークのみ**: `latest`（評価用には使用しない）
- ダイジェスト
  - リリースごとに `ghcr-digests-<X.Y.Z>.txt` を Release assets に添付する
  - **必須**: Jetson/arm64 runtime の `RepoDigests`
  - **あれば追記**: PC/amd64 runtime・Raspberry Pi 付帯サービスの `RepoDigests`
- マルチアーチ
  - 同じタグで `linux/arm64` を必須提供し、`linux/amd64` は発行済みの場合に同タグへ追加する（`docker pull --platform=...` で取得可）

## Compose での固定例（ダイジェスト pin）

Jetson runtime:

```bash
MAGICBOX_IMAGE=ghcr.io/michihitotakami/totton-audio-system@sha256:<jetson-digest> \
  docker compose -f docker/jetson/docker-compose.jetson.runtime.yml up -d
```

Raspberry Pi 付帯サービス:

```bash
USB_I2S_BRIDGE_IMAGE=ghcr.io/michihitotakami/totton-audio-system@sha256:<raspi-bridge-digest> \
RASPI_CONTROL_API_IMAGE=ghcr.io/michihitotakami/totton-audio-system@sha256:<raspi-control-digest> \
  docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
```

> `latest` は動作確認用に残すが、評価版の配布では **必ずタグまたはダイジェスト** を指定する。
