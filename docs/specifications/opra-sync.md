## OPRA Online Sync 仕様（Issue #85）

### 目的
- OPRA データベース（EQ/製品情報）を **オンライン同期**できるようにする。
- 同期は **コミットSHAで固定**できる（= 取得したデータの同一性を担保）。
- Jetson（Docker運用）を想定し、**オフラインでも動作**し、**ロールバック可能**であること。
- 可能であれば **管理画面（Web UI）から更新**できること。

### 非目的（本Issueでは仕様のみ）
- 差分更新アルゴリズムの実装（まずは全量取得で良い）
- OPRAの画像/ロゴ/アートワークの完全ミラー（必要なら別仕様）

---

## 背景 / 現状
- 現状は `data/opra-db` を git submodule として取り込み、`dist/database_v1.jsonl` を参照している。
  - 実装: `scripts/integration/opra.py` の `DEFAULT_OPRA_PATH`
- これだと「リリース時点の固定」「ランタイムでの更新」「オフラインフォールバック」「管理画面更新」が整理しづらい。

---

## 要求事項

### MUST
- **SHA固定**: 同期結果は必ず「どのコミットの `database_v1.jsonl` か」を記録し、同一SHAなら同一内容であることを担保する。
- **原子的切替**: 取得中に既存DBを壊さない。成功したら一瞬で切替（atomic swap）する。
- **ロールバック**: 直前（最低1世代）に戻せる。
- **オフライン**: ネットワーク不通でも、最後に成功したローカルキャッシュで検索/適用が動く。
- **管理API**: Web UI から更新できるためのAPIを提供する（実装は後続Issue）。
- **セキュリティ**: 更新操作は管理者限定（認証必須）。

### SHOULD
- **起動時チェック**（更新の有無確認のみ、適用は手動/設定次第）
- **定期チェック**（systemd timer / 内部スケジューラ）。
- **更新通知**: UIに「更新あり」を表示（ポーリングで十分）。

### MAY
- Cloudflare配信（`http://opra.roonlabs.net/database_v1.jsonl`）の利用。
  - ただし「SHA固定」には不向きなため、**利用する場合でも取得後にSHA/ハッシュを記録し、結果として固定化**する。

---

## データソース（推奨設計）

### 推奨: GitHub Raw + commit SHA pin
- URL 形式:
  - `https://raw.githubusercontent.com/opra-project/OPRA/<COMMIT_SHA>/dist/database_v1.jsonl`
- 特徴:
  - **SHA固定**が最も簡単。
  - 取得対象が単一ファイルで済む（軽量）。
- 課題:
  - GitHub Raw のレート制限/障害。

### 代替: OPRA Cloudflare配信
- URL:
  - `http://opra.roonlabs.net/database_v1.jsonl`
- 特徴:
  - キャッシュされるため安定/高速。
- 課題:
  - 取得時点で「どのコミット相当か」は直接わからない。
  - **仕様としては、取得後にハッシュを保存し“その内容”を固定化**する。

### 代替: GitHub Release Asset（将来）
- OPRA側が正式に Release + asset を出すようになったら採用余地。
- この場合も「tag→commit」「asset sha256」をメタデータとして保存する。

---

## ローカル配置（キャッシュレイアウト）

### 1) 永続ディレクトリ
Docker/Jetsonを想定し、永続化ボリュームに配置する。

- **推奨パス（論理）**: `${DATA_DIR}/opra/`
  - 例（ホスト）: `/var/lib/gpu_upsampler/opra/`
  - 例（Docker）: `/data/opra/`（volume）

### 2) ディレクトリ構成

- `${OPRA_DIR}/current`（シンボリックリンク）
- `${OPRA_DIR}/versions/<commit_sha>/database_v1.jsonl`
- `${OPRA_DIR}/versions/<commit_sha>/metadata.json`
- `${OPRA_DIR}/lock/opra_update.lock`

### 3) metadata.json（例）
- `commit_sha`: 取得元コミットSHA（不明な場合は `unknown`）
- `source`: `github_raw` | `cloudflare` | `manual`
- `source_url`: 実際に取得したURL
- `downloaded_at`: ISO8601
- `sha256`: ファイルハッシュ
- `size_bytes`
- `stats`: `{ vendors, products, eq_profiles }`（簡易検証用）

---

## 同期アルゴリズム（全量取得・原子的切替）

### 手順（成功条件）
1. 更新ロック取得（多重実行防止）
2. 対象バージョン決定
   - `target_ref` が SHA の場合: そのSHAを使用
   - `latest` の場合: GitHub APIで `opra-project/OPRA` の `main` HEAD SHA を取得
3. 一時ファイルへダウンロード
4. 検証
   - JSONLとして最低限 parse できること（先頭N行チェック）
   - 期待する `type`（vendor/product/eq）が含まれること
   - sha256 を算出
5. `${OPRA_DIR}/versions/<commit_sha>/` へ配置
6. `current` を新バージョンへ **atomicに付け替え**
   - 例: `current.tmp` → `rename()`
7. 旧バージョン保持（例: 最新3世代）
8. ロック解放

### 失敗時
- 既存 `current` は維持（サービス継続）。
- UI/ログに失敗理由を出す。

---

## オフラインモード

### 優先順位
1. `${OPRA_DIR}/current/database_v1.jsonl`
2. （開発環境のみ）従来の submodule パス `data/opra-db/dist/database_v1.jsonl`
3. どちらも無い場合は 503（UIに「未インストール」表示）

---

## Web API（管理者向け）

### 認証
- **必須**: 管理APIは認証必須。
  - 最小: APIキー（`X-Admin-Token`）
  - 望ましい: Basic認証 or リバースプロキシ（Nginx）

### エンドポイント案
- `GET /api/opra/sync/status`
  - 現在の `commit_sha`, `downloaded_at`, `stats`, `source` を返す
- `GET /api/opra/sync/available`
  - `latest` の候補（GitHub HEAD SHA / Cloudflare情報）を返す
- `POST /api/opra/sync/update`
  - body: `{ "target": "latest" | "<commit_sha>" , "source": "github_raw" | "cloudflare" }`
  - 返却: ジョブID（非同期実行の場合） or 即時結果
- `POST /api/opra/sync/rollback`
  - 直前バージョンへ切替

### 非同期実行（推奨）
- DBが大きい場合、HTTPリクエストを長時間ブロックしない。
- 方式: バックグラウンドタスク + 状態ファイル（`opra_sync_state.json`）

---

## Web UI（管理画面）

### 表示項目
- 現在のOPRAバージョン
  - `commit_sha` / `downloaded_at` / `vendors/products/eq_profiles`
- 更新状態
  - 「更新あり」「最新」「失敗（理由）」

### 操作
- 「更新チェック」ボタン（available取得）
- 「更新」ボタン
  - `latest` か `commit_sha` を指定可能
- 「ロールバック」ボタン

---

## 更新タイミング（ポリシー）

### デフォルト（推奨）
- **起動時**: 更新有無チェック（自動適用はしない）
- **手動**: 管理画面から適用

### オプション
- 定期チェック（例: 1日1回）
- 自動適用（危険なので明示設定が必要）

---

## 互換性 / 既存実装への影響

### scripts/integration/opra.py
- `DEFAULT_OPRA_PATH` を「submodule固定」から「OPRA_DIR/current を優先」に変更する。
- DB未存在時のエラーメッセージも「submodule update」ではなく「同期/インストール」を案内する。

### routers/opra.py
- 既存の検索/適用APIは基本維持。
- 追加で `opra/sync/*` の管理APIを別routerとして追加する。

---

## 実装タスク分解（後続Issueの雛形）
- OPRAキャッシュディレクトリとメタデータ形式の実装
- GitHub Raw からのSHA固定ダウンロード実装
- atomic swap + ロールバック
- 管理API（認証込み）
- 管理画面UI（表示+操作）
- Docker volume / 設定項目の追加

---

## 参考
- Issue: #85
- OPRA Cloudflare配信: `http://opra.roonlabs.net/`
- OPRA GitHub: `https://github.com/opra-project/OPRA`
