# Docker（評価者向け / ソース不要）

このREADMEは **評価者（Trial）向け**です。**ソースコード不要**で、GHCR の image を pull して起動します。

---

## GHCRイメージ（公開 / アーキ別タグ / ダイジェスト固定）

- パッケージ: `ghcr.io/michihitotakami/totton-audio-system`（公開、認証不要で pull 可能）
- サポートプラットフォーム:
  - `linux/arm64`（Jetson Orin 系）: `MAGICBOX_IMAGE` / `USB_I2S_BRIDGE_IMAGE` / `RASPI_CONTROL_API_IMAGE`
  - `linux/amd64`（開発PC/NVIDIA GPU）: amd64 イメージが発行されたリリースでは、`--platform=linux/amd64` を明示して pull 可能
- タグ運用:
  - リリースタグ: `vX.Y.Z`（推奨） / `X.Y.Z`
  - 評価用ダイジェスト: Release アセットの `ghcr-digests-<X.Y.Z>.txt` を参照し、`@sha256:...` で pin
  - `latest`: スモーク向け。評価時は使わず、必ずバージョンまたはダイジェストを指定
- 例（Jetsonで digest pin する場合）:

```bash
export MAGICBOX_IMAGE=ghcr.io/michihitotakami/totton-audio-system@sha256:<jetson-digest>
docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
```

例（PC/amd64で pull できることを確認する場合。実行には NVIDIA GPU + CUDA ドライバが必要）:

```bash
docker pull --platform=linux/amd64 ghcr.io/michihitotakami/totton-audio-system@sha256:<amd64-digest>
```

---

## Jetson: runtime-only（image-based）

詳細な前提・ログ取得・既知トラブル・ロールバックは、まずこちら:

- [Jetson 評価者向け導入ガイド（ソース不要 / Docker）](../docs/jetson/evaluator_guide_docker.md)
- 評価者オンボーディング（QuickStart / チェックリスト / 報告テンプレ）:
  - [評価者向け QuickStart（オンボーディング）](../docs/jetson/evaluator_quickstart.md)

最短コマンドだけ再掲:

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
docker compose -f jetson/docker-compose.jetson.runtime.yml logs -f
```

Web UI の公開先について（重要）:

- runtime compose のデフォルトは **USB gadget (`192.168.55.1`) のみ**に bind します（意図しないWi-Fi/LAN露出を避けるため）。
- USB gadget を使わず LAN で公開したい場合は、明示的に上書きしてください（インターネット公開は禁止/非推奨）。

```bash
cd docker
MAGICBOX_PUBLISH_IP=0.0.0.0 docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
```

停止:

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml down
```

---

## Raspberry Pi: runtime-only（image-based）

Raspberry Pi 側は **USB/UAC2入力 → I2S出力（usb-i2s-bridge）**がデフォルトです。

最短コマンド:

```bash
cd /path/to/magicbox-root
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
```

停止:

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml down
```

設定/デバイス確認/トラブルシュート:

- `raspberry_pi/README.md`
