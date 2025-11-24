# API Changelog

APIへの重要な変更はこのファイルに記録されます。

フォーマットは [Keep a Changelog](https://keepachangelog.com/ja/1.0.0/) に基づきます。

## [Unreleased]

### Added
- **DAC**: DAC Capability検出API (#199)
  - `GET /dac/capabilities` - DAC対応サンプリングレート取得
  - `GET /dac/devices` - デバイス一覧とCapability
  - `GET /dac/supported-rates` - レートファミリ（44k/48k）別対応レート
  - `GET /dac/max-ratio` - 最大アップサンプリング倍率取得
  - `GET /dac/validate-config` - 設定バリデーション

### Security
- ALSAデバイス名のバリデーション追加（パストラバーサル防止）

## [1.0.0] - 2024-11-24

### Added
- 初期APIリリース
- **Status**: システム状態・設定管理
  - `GET /status` - システム状態取得
  - `POST /settings` - 設定更新
  - `GET /devices` - ALSAデバイス一覧
  - `WS /ws/stats` - リアルタイム統計
- **Daemon**: デーモン制御
  - `POST /daemon/start|stop|restart` - ライフサイクル管理
  - `GET /daemon/status` - デーモン状態
  - `GET /daemon/zmq/ping` - ZeroMQ疎通確認
- **EQ**: プロファイル管理
  - CRUD操作（list, import, activate, delete）
  - プロファイル検証機能
- **OPRA**: ヘッドホンEQデータベース連携
  - 検索・フィルタ機能
  - Modern Target (KB5000_7) 補正対応

### Deprecated
- `POST /restart` - `POST /daemon/restart` を使用してください

---

## テンプレート

新しいバージョンを追加する際は以下のフォーマットを使用してください：

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- 新しいエンドポイント

### Changed
- 既存エンドポイントの変更

### Deprecated
- 非推奨になったエンドポイント

### Removed
- 削除されたエンドポイント

### Fixed
- バグ修正

### Security
- セキュリティ修正
```
