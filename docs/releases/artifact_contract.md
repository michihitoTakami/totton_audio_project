# 配布物仕様（Artifact Contract）

Issue: #1053 / Epic: #1051

## 目的

評価者が **ソースコード不要** で Magic Box を試せるように、GitHub Release に載せる成果物（アセット）と検証手順を **固定化** する。

本ドキュメントは「何を配布するか」「どう検証するか」「外部データ（OPRA/HUTUBS 等）を同梱するか」を明確にし、運用・自動化（CI/OTA）に接続できるようにする。

---

## スコープ

本 Artifact Contract は、GitHub Release（tag: `v*`）で提供する以下を対象とする：

- **実行可能な配布パッケージ（tarball）**
- **OpenAPI 仕様（json）**
- **SHA256（整合性チェック）**
- **署名（改ざん検知 / 将来含む）**
- **SBOM（ソフトウェア部品表）**

---

## バージョニング（Release タグ）

- Release は `vX.Y.Z`（例: `v1.2.3`）を使用する
- 各成果物には同一の `VERSION` を埋め込み、相互に整合していること
  - tarball 内 `manifest.json`
  - Release asset 名
  - `.sha256` / `checksums.sha256`
  - SBOM

---

## Release に載せる成果物（Assets）

### 1) 配布パッケージ（tarball）

**MUST（必須）**

- `magicbox-update-${VERSION}-jetson-arm64.tar.gz`

**SHOULD（推奨）**

- `magicbox-update-${VERSION}-pc-amd64.tar.gz`（開発/検証用。運用必須ではない）

> Jetson（arm64）は実機評価の主戦場のため **必須**。PC（amd64）は開発者向けであり **任意** とする。

#### tarball の最低要件（ディレクトリ構成）

`docs/jetson/deployment/ota-update.md` のパッケージ形式に合わせる。

例：

```
magicbox-update-${VERSION}-<platform>.tar.gz
├── manifest.json
├── checksums.sha256
├── LICENSE
├── LICENSE.ja.md
├── NOTICE.md
├── THIRD_PARTY_LICENSES.md
├── bin/
├── lib/
├── web/
└── scripts/
```

> `LICENSE` の条件（非商用再配布時に `LICENSE` / `NOTICE.md` / `THIRD_PARTY_LICENSES.md` を同梱する等）に整合させるため、tarball にはライセンス/帰属ファイルを **必ず含める**。

#### manifest.json（最低要件）

`manifest.json` は少なくとも以下を含む：

- `version`: `${VERSION}`
- `min_compatible`: 互換性下限（例: `1.0.0`）
- `platform`: `jetson-arm64` | `pc-amd64`
- `git_sha`: ビルド元コミット SHA
- `build_time`: ISO8601
- `components`: 配下ファイル/ディレクトリと `sha256`（`sha256:<hex>` 形式）
- `third_party_data_policy`: 後述（OPRA/HUTUBS）

> `manifest.json` の具体例は `docs/jetson/deployment/ota-update.md` を参照。

---

### 2) OpenAPI（機械可読な仕様）

**MUST（必須）**: Release assets として同梱する（tarball 内同梱でもよいが、Release asset を正とする）

- `openapi.json`（Jetson Web API / OpenAPI 3.1）
- `raspi_openapi.json`（Pi Control API / OpenAPI 3.1）

参照：
- `docs/api/README.md`
- `docs/api/openapi.json`
- `docs/api/raspi_openapi.json`

---

### 3) SHA256（整合性チェック）

**MUST（必須）**

- 各 Release asset につき、同名の `.sha256` を付与する
  - 例: `magicbox-update-${VERSION}-jetson-arm64.tar.gz.sha256`
  - 例: `openapi.json.sha256`
  - 例: `sbom-${VERSION}-jetson-arm64.cdx.json.sha256`

`.sha256` の中身は `sha256sum` 互換（`<hash>  <filename>`）とする。

**SHOULD（推奨）**

- `checksums.sha256`（Release 全アセットのまとめ）を追加する

---

### 3.5) 署名（改ざん検知）

**SHOULD（推奨）**

- tarball に対する **detached signature** を Release assets として提供する
  - `magicbox-update-${VERSION}-jetson-arm64.tar.gz.sig`
  - `magicbox-update-${VERSION}-pc-amd64.tar.gz.sig`（PC を出す場合）
- 署名を提供する場合は、検証に必要な **公開鍵情報** を明示する
  - 例: Release の本文に公開鍵 fingerprint を記載
  - 例: Release assets として公開鍵（`.asc` 等）を添付

検証例（概念）:

- `gpg --verify magicbox-update-${VERSION}-jetson-arm64.tar.gz.sig magicbox-update-${VERSION}-jetson-arm64.tar.gz`

> 署名の扱いは `docs/jetson/deployment/ota-update.md` の「セキュリティ / パッケージ署名（将来）」を参照。

---

### 4) SBOM（Software Bill of Materials）

**MUST（必須）**

- tarball（platform）ごとに SBOM を 1 つ提供する（Release assets）
  - `sbom-${VERSION}-jetson-arm64.cdx.json`
  - `sbom-${VERSION}-pc-amd64.cdx.json`（PC を出す場合）

**SBOM フォーマット（固定）**

- **CycloneDX JSON**（拡張子: `.cdx.json`）
- 生成対象は tarball の内容（バイナリ/ライブラリ/Web/UI/同梱スクリプト等）

> 将来 Docker image 配布へ移行した場合は、image 単位の SBOM（別名）を追加する。

---

## Jetson（arm64）と PC（amd64）の扱い

- **Jetson（arm64）**: MUST（必須）
  - 目的: 実運用/評価のデフォルトターゲット
  - 成果物: `magicbox-update-${VERSION}-jetson-arm64.tar.gz` + 付随ファイル
- **PC（amd64）**: SHOULD（推奨、ただし任意）
  - 目的: 開発者のローカル検証、CI/テスト
  - 成果物: `magicbox-update-${VERSION}-pc-amd64.tar.gz` + 付随ファイル

---

## Raspberry Pi ブリッジ（UAC2 入力の受け口）

本プロジェクトの「I/O 分離構成」では、**Raspberry Pi が UAC2 の受け口（入力ハブ）**となり、Jetson は **RTP 受信 + GPU 処理**に専念する。

このとき「Pi 側の UAC2 としての受け口」は、OS 上では **ALSA の入力デバイス**として扱い、`raspberry_pi/usb_i2s_bridge` がそのデバイスから rate/format を取得して下流（I2S / RTP）へ流す。

### 受け口の定義（MUST）

- **受け口（UAC2 input）**: Pi 上の ALSA デバイス（`arecord` で開けるデバイス）
  - `raspberry_pi/usb_i2s_bridge` の `USB_I2S_CAPTURE_DEVICE` が参照するものを **UAC2 input** とみなす
  - デバイス名は環境依存（カード番号が変わるため）
    - 例: `hw:3,0`
    - 例: `hw:CARD=UAC2Gadget,DEV=0`（典型例。実体は `/proc/asound/cards` / `arecord -l` で確認）

### 期待仕様（SHOULD）

- **2ch**、44.1k 系/48k 系のレート切替に追従できること
- 量子化は **24-in-32（`S32_LE`）を推奨**
- レート/フォーマットは **変換せずにパススルー優先**（`USB_I2S_PASSTHROUGH=true` を想定）

### 関連

- Pi 側ブリッジ/制御 API: `raspberry_pi/docker-compose.yml`（`raspi_openapi.json`）
- 運用/セットアップ（Pi 側 UAC2 受け口の説明）: `docs/setup/pi_bridge.md`

---

## 第三者データ（OPRA / HUTUBS 等）の方針

### OPRA（ヘッドホンEQデータベース）

**結論: 同梱しない（初回取得 / キャッシュ運用）**

- 理由:
  - データ更新頻度が高く、固定同梱は運用コストが高い
  - 配布サイズ増加を避けたい
  - 取得時点を SHA/ハッシュで固定した方が再現性を担保しやすい

実装・運用の前提は `docs/specifications/opra-sync.md` を参照。

### HUTUBS（Crossfeed/HRTF データ）

**結論: 生データ（SOFA 等）は同梱しない**

- 生データは外部から取得する（開発者向け）
- 配布物として必要な場合は、生データではなく **派生物（生成済み係数/メタデータ）** を同梱する
  - 例: Crossfeed 用の係数・メタ情報（利用する場合）

> 目的は「評価者がソース不要で試す」ことなので、評価に必要な最低限の派生物は tarball に含める運用を推奨する（生データの再配布はしない）。

---

## 検証手順（利用者向け）

### 1) SHA256 検証（必須）

- `.sha256` をダウンロードし、`sha256sum -c` で検証できる形式で提供する
- 例（概念）:
  - tarball と `.sha256` を同じディレクトリに置き、`sha256sum -c <asset>.sha256`

### 2) OpenAPI の参照

- Release asset の `openapi.json` / `raspi_openapi.json` を正として参照する
- 参考: `docs/api/README.md`

### 3) SBOM の参照

- `sbom-${VERSION}-<platform>.cdx.json` を確認し、含まれる依存関係/ライセンス/メタ情報を監査できること

---

## 関連ドキュメント

- `docs/jetson/deployment/ota-update.md`（パッケージ形式・OTA検証）
- `docs/api/README.md`（OpenAPI 配布と更新）
- `docs/setup/pi_bridge.md`（Raspberry Pi ブリッジ / UAC2 受け口）
- `docs/specifications/opra-sync.md`（OPRA 初回取得/キャッシュ方針）
- `docs/releases/public_repo.md`（公開範囲/配布物に含めるものの前提）
