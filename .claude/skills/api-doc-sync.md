# API Documentation Sync

FastAPI変更後のOpenAPI同期とCHANGELOG更新を自動化します。

## Description

このSkillは、FastAPIエンドポイント変更時に、OpenAPI仕様の自動生成とCHANGELOG更新を行います。pre-commitフックとの統合により、APIドキュメントの整合性を保ちます。

## Trigger Words

- `sync api docs`
- `update openapi`
- `api changelog`
- `API同期`
- `ドキュメント更新`

## Requirements

- `uv` がインストールされていること
- FastAPI アプリケーションが正常に起動すること
- `web/` ディレクトリに変更があること
- 実行ディレクトリ: プロジェクトルート

## Execution Steps

```bash
# 1. 変更ファイル検出
git diff --name-only | grep -E '^web/.*\.py$'

# 2. OpenAPI仕様生成
uv run python scripts/integration/export_openapi.py

# 3. 差分確認
git diff docs/api/openapi.json

# 4. 変更エンドポイント解析
# - 新規追加: POST /eq/import
# - 変更: GET /status に crossfeed_enabled フィールド追加
# - 削除: なし

# 5. CHANGELOG自動更新
# docs/api/CHANGELOG.md に新しいエントリ追加

# 6. 検証
uv run python scripts/integration/export_openapi.py --check

# 7. コミット準備
git add docs/api/openapi.json docs/api/CHANGELOG.md
```

## Parameters

### オプションパラメータ

- **breaking_change**: 破壊的変更フラグ（true/false）
  - 例: `sync api docs breaking`
  - デフォルト: false

## Expected Output

### 変更あり:
```markdown
# API ドキュメント同期完了

## 変更されたファイル
- web/routers/eq.py
- web/routers/daemon.py
- web/models.py

## OpenAPI変更内容

### 新規エンドポイント
- `POST /eq/import` - Import EQ profile from file
  - リクエスト: multipart/form-data
  - レスポンス: EQProfile

### 変更されたエンドポイント
- `GET /status` - Added `crossfeed_enabled` field
  - 新フィールド: crossfeed_enabled (boolean)
- `POST /api/status/settings` - Added `crossfeed_level` parameter
  - 新パラメータ: crossfeed_level (0.0-1.0)

### 削除されたエンドポイント
（なし）

## CHANGELOG更新
新しいエントリを追加しました:

\`\`\`markdown
## [1.2.0] - 2025-12-06

### Added
- POST /eq/import - Import EQ profile from file
- GET /status - Added crossfeed_enabled field

### Changed
- POST /api/status/settings - Added crossfeed_level parameter (0.0-1.0)
\`\`\`

## 次のステップ
- [ ] `docs/api/CHANGELOG.md` の説明文をレビュー
- [ ] 破壊的変更がある場合は、バージョン番号を確認
- [ ] コミット: `git commit -m "docs(api): Update OpenAPI spec and CHANGELOG"`
```

### 変更なし:
```markdown
# API ドキュメント同期

⚠️ `web/` ディレクトリに変更が検出されませんでした。

## 確認事項
- web/routers/ にエンドポイント変更がありますか？
- web/models.py に Pydantic モデル変更がありますか？

変更がある場合は、まずコミットしてから再実行してください。
```

## Error Handling

このSkillはベストエフォート戦略を採用しています：

1. **OpenAPI生成失敗時**:
   - FastAPIアプリのエラー詳細表示
   - スタートアップエラーの原因を特定
   - 修正すべきファイルを提示
   - 例: "ModuleNotFoundError: No module named 'pydantic_core'"

2. **CHANGELOG競合時**:
   - 手動編集を促す
   - テンプレートを提供
   - 既存エントリとの統合方法を説明

3. **web/変更なし時**:
   - 警告表示
   - 実行をスキップ
   - 変更が必要な理由を説明

4. **検証失敗時**:
   - `export_openapi.py --check` の出力を表示
   - OpenAPIスキーマの不整合を指摘
   - 修正方法を提案

## Best Practices

### API変更時のチェックリスト

このSkillは以下をチェックします：

- [ ] `response_model` が全エンドポイントに指定されているか
- [ ] HTTPExceptionが適切に定義されているか
- [ ] 適切なタグ（tags）が設定されているか
- [ ] 非推奨エンドポイントに `deprecated=True` が付いているか

### pre-commitフックとの統合

このSkillは `pre-commit` フックと統合されています：

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: openapi-export
      name: Export OpenAPI spec
      entry: uv run python scripts/integration/export_openapi.py
      files: ^web/.*\.py$
      language: system
```

変更が `web/` ディレクトリにある場合、自動的に実行されます。

### CHANGELOGエントリ形式

```markdown
## [バージョン] - YYYY-MM-DD

### Added
- 新規エンドポイント

### Changed
- 既存エンドポイントの変更

### Deprecated
- 非推奨化されたエンドポイント

### Removed
- 削除されたエンドポイント

### Fixed
- バグ修正
```

## Related Skills

- `worktree-pr-workflow`: API変更時は併用推奨
- `build-and-test`: APIテスト実行

## Implementation Notes

このSkillは以下の既存ツールを活用します：

- `scripts/integration/export_openapi.py`: OpenAPI生成スクリプト
- `docs/api/CHANGELOG.md`: 手動管理されるCHANGELOG
- `docs/api/openapi.json`: 自動生成されるOpenAPI仕様
- `pre-commit`: web/変更時の自動実行

### エンドポイント解析ロジック

1. **新規**: 旧OpenAPIに存在しない、新OpenAPIに存在するパス
2. **変更**: 両方に存在し、スキーマが異なるパス
3. **削除**: 旧OpenAPIに存在し、新OpenAPIに存在しないパス

## Automation Level

**完全自動実行**: OpenAPI生成とCHANGELOGテンプレート作成を自動化します。
ユーザーはCHANGELOG の説明文を確認・編集してください。
