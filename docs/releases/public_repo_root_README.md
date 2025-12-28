# Magic Box Project - Public Repo

このリポジトリは、評価者が参照する **公開用スナップショット**です。

- コア実装（`src/` / `include/`）と内部ツール（`scripts/`）は **含みません**
- 公開する範囲は `docs/releases/public_repo.md` の方針に従います

---

## QuickStart（まずここ）

- Jetson: **評価者向け QuickStart**: `docs/jetson/evaluator_quickstart.md`
- Raspberry Pi（USB/UAC2 → I2S / RTPフォールバック）: `raspberry_pi/README.md`

---

## Web UI / API 仕様

- OpenAPI: `docs/api/README.md`
  - `docs/api/openapi.json`（Jetson Web/API）
  - `docs/api/raspi_openapi.json`（Pi Control API）

---

## 配布（runtime-only / build無し）

- Jetson compose: `docker/jetson/docker-compose.jetson.runtime.yml`
- Raspberry Pi compose: `raspberry_pi/docker-compose.raspberry_pi.runtime.yml`

---

## セキュリティ（LAN前提）

- `docs/jetson/security_baseline.md`

---

## ライセンス/帰属

- `LICENSE` / `LICENSE.ja.md`
- `NOTICE.md`
- `THIRD_PARTY_LICENSES.md`
