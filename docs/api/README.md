# GPU Upsampler Web API

GPU音声アップサンプラーのREST API仕様です。

## 概要

| 項目 | 値 |
|------|-----|
| ベースURL | `http://localhost:11881` |
| 認証 | なし（ローカルネットワーク専用） |
| フォーマット | JSON |

## インタラクティブドキュメント

サーバー起動後、以下のURLでインタラクティブなAPIドキュメントにアクセスできます：

- **Swagger UI**: http://localhost:11881/docs
- **ReDoc**: http://localhost:11881/redoc

## エンドポイント一覧

### Status (`/status`)
システム状態と設定の管理

| Method | Path | 説明 |
|--------|------|------|
| GET | `/status` | システム状態取得 |
| POST | `/settings` | 設定更新 |
| GET | `/devices` | ALSAデバイス一覧 |
| WS | `/ws/stats` | リアルタイム統計（WebSocket） |

### Daemon (`/daemon`)
オーディオデーモンのライフサイクル管理

| Method | Path | 説明 |
|--------|------|------|
| POST | `/daemon/start` | デーモン起動 |
| POST | `/daemon/stop` | デーモン停止 |
| POST | `/daemon/restart` | デーモン再起動 |
| GET | `/daemon/status` | デーモン状態取得 |
| GET | `/daemon/zmq/ping` | ZeroMQ疎通確認 |
| POST | `/daemon/zmq/command/{cmd}` | ZeroMQコマンド送信 |

### EQ (`/eq`)
EQプロファイル管理

| Method | Path | 説明 |
|--------|------|------|
| GET | `/eq/profiles` | プロファイル一覧 |
| POST | `/eq/validate` | プロファイル検証 |
| POST | `/eq/import` | プロファイルインポート |
| POST | `/eq/activate/{name}` | プロファイル有効化 |
| POST | `/eq/deactivate` | EQ無効化 |
| DELETE | `/eq/profiles/{name}` | プロファイル削除 |
| GET | `/eq/active` | アクティブプロファイル取得 |

### OPRA (`/opra`)
OPRAヘッドホンEQデータベース連携

| Method | Path | 説明 |
|--------|------|------|
| GET | `/opra/stats` | データベース統計 |
| GET | `/opra/vendors` | ベンダー一覧 |
| GET | `/opra/search` | ヘッドホン検索 |
| GET | `/opra/products/{id}` | 製品詳細取得 |
| GET | `/opra/eq/{id}` | EQプロファイル取得 |
| POST | `/opra/apply/{id}` | EQプロファイル適用 |

### DAC (`/dac`)
DAC Capability検出とサンプリングレートフィルタリング

| Method | Path | 説明 |
|--------|------|------|
| GET | `/dac/capabilities` | DAC対応サンプリングレート取得 |
| GET | `/dac/devices` | デバイス一覧とCapability |
| GET | `/dac/supported-rates` | レートファミリ別対応レート |
| GET | `/dac/max-ratio` | 最大アップサンプリング倍率 |
| GET | `/dac/validate-config` | 設定バリデーション |

## レスポンス形式

### 成功レスポンス（変更系）
```json
{
  "success": true,
  "message": "操作完了メッセージ",
  "data": { ... },
  "restart_required": false
}
```

### エラーレスポンス
```json
{
  "detail": "エラーメッセージ",
  "error_code": "HTTP_404"
}
```

## OpenAPI仕様

機械可読なOpenAPI仕様は以下のファイルで提供されます：

- **[openapi.json](./openapi.json)** - OpenAPI 3.1仕様（JSONフォーマット）

### OpenAPIファイルの更新

`web/` 配下のPythonファイルを変更すると、pre-commitフックにより自動的に `openapi.json` が更新されます。

手動で更新する場合：
```bash
uv run python scripts/export_openapi.py
```

最新かどうか確認する場合：
```bash
uv run python scripts/export_openapi.py --check
```

## ライセンス

OPRAデータは [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) ライセンスです。
