# API Changelog

APIへの重要な変更はこのファイルに記録されます。

フォーマットは [Keep a Changelog](https://keepachangelog.com/ja/1.0.0/) に基づきます。

## [Unreleased]

### Added
- **Phase Type Control**: 位相タイプのランタイム切り替え（フィルタは常時プリロード）
  - `GET /daemon/phase-type` - 現在の位相タイプ取得
  - `PUT /daemon/phase-type` - 位相タイプ変更（即時反映）
- **DAC**: DAC Capability検出API (#199)
  - `GET /dac/capabilities` - DAC対応サンプリングレート取得
  - `GET /dac/devices` - デバイス一覧とCapability
  - `GET /dac/supported-rates` - レートファミリ（44k/48k）別対応レート
  - `GET /dac/max-ratio` - 最大アップサンプリング倍率取得
  - `GET /dac/validate-config` - 設定バリデーション
- **Low-Latency Validation Toolkit** (#355)
  - `scripts/analysis/inspect_impulse.py` に partition summary / latency 推定を追加
  - `scripts/analysis/verify_frequency_response.py` に fast/tail スペクトル比較と自動スキップ機能を追加
  - `docs/investigations/low_latency_partition_validation.md` にループバック手順とQA基準を掲載

### Changed
- **FastAPI Lifespan Migration**: `lifespan` コンテキストマネージャで RTPautostart/シャットダウンを管理し、TCPテレメトリポーラーを削除
- **Protocol Cleanup**: `tcp_input` ルーターとモデルを削除し、Web API は RTP入力経路に一本化

### Removed
- **TCP Input API & UI** (#846)
  - `GET /api/tcp-input/status`
  - `POST /api/tcp-input/start`
  - `POST /api/tcp-input/stop`
  - `PUT /api/tcp-input/config`
  - `/tcp-input` ページおよび `web/static/js/tcp_input.js`, `web/templates/pages/tcp_input.html`
  - ZeroMQ コマンド `TCP_INPUT_STATUS`, `TCP_INPUT_START`, `TCP_INPUT_STOP`, `TCP_INPUT_CONFIG_UPDATE`

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
