# Raspberry Pi ブリッジ（UAC2 受け口）

このドキュメントは、Raspberry Pi を **「UAC2 の受け口」**として使い、入力を下流（I2S / RTP）へ流すための要点をまとめます。

対象コンポーネント：

- `raspberry_pi/usb_i2s_bridge`（USB入力 → I2S 出力のブリッジ）
- `raspberry_pi/control_api.py`（Pi 側の制御 API。OpenAPI は `docs/api/raspi_openapi.json`）

---

## 受け口（UAC2 input）の定義

Pi 側の「UAC2 の受け口」は **ALSA の入力デバイス**として扱います。

- `USB_I2S_CAPTURE_DEVICE` が参照する ALSA デバイスを **UAC2 input** とみなします
  - 例: `hw:3,0`
  - 例: `hw:CARD=UAC2Gadget,DEV=0`（環境によりカード番号は変動）

確認方法（例）：

- `arecord -l`
- `cat /proc/asound/cards`

> `hw:X,Y` の X はカード番号で、USBの抜き差し等で変わることがあります。安定運用したい場合は `hw:CARD=<id>,DEV=<n>` 形式を優先してください。

---

## 期待仕様（推奨）

- **2ch**（ステレオ）
- 44.1k 系/48k 系のレート切替に追従できること
- 量子化は **24-in-32（`S32_LE`）推奨**
- **変換は行わず**（橋渡しに特化）入力の rate/format をそのまま下流へ流す

これらは `raspberry_pi/usb_i2s_bridge/bridge.py` が「入力の rate/format を検出して追従する」前提に合わせたものです。

---

## 起動（Docker / Compose）

Pi 上で以下を起動します（デフォルトは I2S ブリッジ構成）。

### 評価者向け（ソース不要 / runtime-only）

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml up -d
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs -f
```

停止:

```bash
docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml down
```

### 開発者向け（ローカルビルド）

```bash
docker compose -f raspberry_pi/docker-compose.yml up -d --build
```

起動サービス：

- `usb-i2s-bridge`（`/dev/snd` が必要）
- `raspi-control-api`（既定で `usb0` に bind、ポート `8081`）

---

## 設定（config.env）

設定は `USB_I2S_CONFIG_PATH`（既定: `/var/lib/usb-i2s-bridge/config.env`）に集約されます。
初回起動時は `raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env` が seed としてコピーされます。

最低限、次を **環境に合わせて** 調整します：

- `USB_I2S_CAPTURE_DEVICE`: UAC2 input（受け口）
- `USB_I2S_PLAYBACK_DEVICE`: I2S 出力（典型: `hw:0,0`）
- `USB_I2S_FALLBACK_RATE`: `44100`（`hw_params` 未確定時のフォールバック）
- `USB_I2S_PREFERRED_FORMAT`: `S32_LE`（推奨）

---

## 補足: UAC2 gadget（PC から Pi へ音を入れる場合）

UAC2 “受け口” を **USB Gadget Mode（UAC2）**で提供する場合、Pi 側に UAC2 gadget を構成し、上で説明した ALSA デバイスとして見える必要があります。

本リポジトリでは、この領域は `docs/roadmap.md` の Phase 3.1 に記載されており、設定の自動化（ConfigFS スクリプト等）は今後の整備対象です。
