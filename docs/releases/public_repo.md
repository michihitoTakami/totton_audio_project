# Public Repo 方針（Private → Public 分離）

Issue: #1052 / Epic: #1051

## 目的

- `src/` と `scripts/`（データプレーン/内部ツール）を **非公開** のまま、外部に共有したい範囲（docs / OpenAPI / UI / 配布定義 / ライセンス）を **Public Repo** として切り出せるようにする。
- 「何を公開する/しない」を曖昧にせず、Private→Public の同期（更新）手段を **運用として成立** させる。

## 非目的（やらないこと）

- GPU Audio Engine（`src/`, `include/`）の公開
- フィルタ生成や解析など、内部ツール（`scripts/`）の公開
- 公開Repoだけで “箱をビルドできる” 状態にすること（Private側のビルド手順/環境依存の私物化を Public に持ち込まない）

---

## 公開する範囲（Public Repo に含める）

Public Repo は **Control Plane / UI / 配布定義 / ドキュメント** を中心に、次を含める。

### 1) ドキュメント

- `docs/api/**`（OpenAPI json と API README）
- `docs/jetson/**`（運用/ネットワーク/トラブルシュート）
- `docs/releases/**`（リリースノート）
- `docs/roadmap.md`
- `docs/architecture/**`（アーキテクチャ説明）
- `docs/specifications/opra-sync.md`（OPRA 初回取得/キャッシュ方針。運用/仕様の範囲として公開OK）

> 補足: `docs/setup/**` や `docs/dev/**` は、Private側の実装/ビルドに深く依存するものが多いため **Public Repo には含めない**（後述）。

### 2) OpenAPI（機械可読な仕様）

- `docs/api/openapi.json`
- `docs/api/raspi_openapi.json`

Public Repo では **生成物（json）を正** とし、生成スクリプト（`scripts/integration/...`）は Private に残す。

### 3) UI / Web 管理画面

- `web/templates/**`（Jinja2 テンプレート）
- `web/static/**`（CSS/JS）
- `web/main.py` と `web/routers/**`（APIルーティング/仕様のソース）

> Note: delimiter の UI ページ（`web/templates/pages/delimiter.html`）は現時点では公開不要のため、public 同期では除外する（Issue #1057）。

### 4) 配布定義（compose / Docker / ランタイム設定）

- `docker/**`（Jetson 等の Compose / Dockerfile / README）
- `raspberry_pi/**`（Pi側の Compose / 設定 / README）

### 5) ライセンス/帰属/公開に必要なメタ情報

- `LICENSE`, `LICENSE.ja.md`
- （必要に応じて）`NOTICE` / `THIRD_PARTY_NOTICES.md` / 帰属ファイル

---

## 公開しない範囲（Public Repo に含めない）

**絶対に Public Repo に入れない**（= export で落とす/同期に含めない）対象：

### 1) 実装コア（非公開）

- `src/**`
- `include/**`

### 2) 内部ツール（非公開）

- `scripts/**`

### 3) 秘密情報・環境依存

- `.env*`, `*.key`, `*.pem`, `*.p12`, `*.crt`
- 社内/個人のインフラ情報（IP/ホスト名/トークン/認証情報）
- CIシークレット、クラウド資格情報

### 4) 配布してはいけない/配布不要なデータ

- `data/coefficients/**`（FIR係数バイナリなど、配布形態が未確定な生成物）
- `data/opra-db/**`（同期キャッシュ/データベース本体。Publicには「取得手段/帰属/利用条件」だけを置く）
- `build/**`（ビルド生成物）
- `test_data/**`（もし存在する場合）

### 5) “ビルド手順の私物化” に該当するドキュメント

Public Repo は “利用/運用/仕様” を目的とし、Private実装に直結する手順は含めない。

- `docs/setup/build.md`
- `docs/setup/pc_development.md`
- `docs/dev/**`
- （状況により）`docs/specifications/**` のうち、内部実装/モデル/係数生成に直結するもの

---

## Private → Public 同期方針（決定）

**結論: Private Repo を正とし、GitHub Actions（CI）で “allowlist export” を Public Repo に push する。**

理由：

- 人手同期は漏れ（機密/過剰公開）が起きやすい
- allowlist によって「入れて良いもの」しか出ないため、安全性が高い
- Public Repo の履歴を **スナップショット（squash）** にでき、Privateの履歴・ファイル名・試行錯誤を漏らしにくい

### 同期の粒度

- **推奨（初期運用）**: **リリースタグ時のみ** Public を更新する
  - `v*` タグの push、または GitHub Release の `published` をトリガーにする
  - 目的: 「公開して良い状態」を人間が宣言したタイミングだけ同期し、無駄な実行・枠消費・事故（意図しない公開）を避ける
- **代替案**: **prod（production）へデプロイ成功したタイミングのみ** Public を更新する
  - 前提: 既存のデプロイパイプラインが GitHub の `deployment` / `deployment_status`（environment=production）を発行できる、または `workflow_run` で “deploy workflow 成功” を検知できる
  - 目的: 「本番に出たもの＝公開して良いもの」という運用に揃える
- Public のコミットメッセージは以下に統一：
  - `Public export from private@<SHA>`

### 同期のやり方（概要）

Private Repo の GitHub Actions で以下を行う：

1. Private `main` を checkout
2. allowlist のパスだけを `public_export/` にコピー（rsync等）
3. denylist（秘密情報/生成物/キャッシュ）を削除
4. `docs/api/*.json` が最新であることを検証（必要なら Private 内で生成して含める）
5. ざっくりでも良いので検査を通す
   - secret scan（パターン/サイズ/拡張子）
   - ライセンス/帰属ファイルの存在チェック
6. Public Repo に bot トークンで push（squash運用）

> Public Repo 側には export スクリプト自体は置かない（`scripts/` を Public に入れない方針のため）。

---

## 公開Repo側の “受け入れ条件” チェックリスト（運用）

同期前に、最低限これだけは満たす：

- [ ] allowlist/denylist が方針通りになっている（`src/`, `scripts/`, 秘密情報が落ちている）
- [ ] `docs/api/openapi.json` と `docs/api/raspi_openapi.json` が最新
- [ ] `LICENSE*` が Public に含まれている
- [ ] OPRA 等の外部データは **データ本体ではなく** 帰属/利用条件/取得手段が明記されている
