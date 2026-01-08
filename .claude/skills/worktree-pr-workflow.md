# Worktree PR Workflow

Issue番号からworktree作成→実装→コミット→PR作成までを自動化します。

## Description

このSkillは、GitHub Issue番号を指定するだけで、worktree作成からPR作成までの標準的な開発フローを自動化します。プロジェクトのGit Workflow制約（mainブランチ直接作業禁止、Issue番号必須）を徹底します。

## Trigger Words

- `create worktree`
- `start feature`
- `new branch`
- `worktree作成`
- `新しい機能`

## Requirements

- `gh` CLI がインストール・認証済みであること
- Git repository: `/home/michihito/Working/totton_audio/`
- GitHub Issue が作成済みであること
- 実行ディレクトリ: プロジェクトルート

## Parameters

### 必須パラメータ

- **issue_number**: GitHub Issue番号（例: 567）
  - `create worktree for #567`
  - `start feature 567`

### オプションパラメータ

- **branch_type**: ブランチタイプ
  - `feature` (デフォルト)
  - `fix`
  - 例: `create worktree for #567 fix`

## Execution Steps

```bash
# 1. Issue情報取得
gh issue view {issue_number} --json title,body

# 2. Issueタイトルからブランチ名用スラッグ生成
# 例: "Implement Claude Skills" → "claude-skills"

# 3. Worktree作成
cd /home/michihito/Working/gpu_os
git worktree add worktrees/{issue_number}-{slug} -b {branch_type}/#{issue_number}-{slug}

# 4. 作成完了メッセージ表示
# （ユーザーはここで実装を行う）

# 5. コミット準備完了時（ユーザーが実装終了後）
cd worktrees/{issue_number}-{slug}
git status
git diff

# 6. コミット（pre-commit自動実行）
git add .
git commit

# 7. プッシュ（pre-push自動実行）
git push -u origin {branch_type}/#{issue_number}-{slug}

# 8. PR作成
gh pr create --title "#{issue_number} {title}" --body "Closes #{issue_number}

{issue_body}

## 変更内容
- [実装した内容を記載]

## テスト
- [ ] ビルド成功
- [ ] 全テスト通過

🤖 Generated with [Claude Code](https://claude.com/claude-code)
"
```

## Expected Output

### Worktree作成時:
```markdown
# Worktree作成完了

## Worktree情報
- パス: `/home/michihito/Working/totton_audio/worktrees/567-claude-skills`
- ブランチ: `feature/#567-claude-skills`
- ベースコミット: dd1a058

## Issue情報
- #567: Implement Claude Skills for project automation
- ラベル: enhancement
- マイルストーン: Phase 2

## 次のステップ
1. 以下のコマンドで worktree に移動してください:
   ```bash
   cd /home/michihito/Working/totton_audio/worktrees/567-claude-skills
   ```

2. 実装を行ってください

3. 実装完了後、コミット・プッシュ・PR作成を実行します
```

### PR作成時:
```markdown
# PR作成完了

## Pull Request
- URL: https://github.com/michihitoTakami/totton_audio/pull/568
- タイトル: #567 Implement Claude Skills for project automation
- ブランチ: feature/#567-claude-skills → main

## 自動実行されたチェック
- ✅ pre-commit hooks passed
- ✅ pre-push tests passed
- ⏳ GitHub Actions running...

## 次のステップ
- PR URLでレビューを待つ
- マージはユーザーがレビュー後に実行
- マージ完了後、worktree を削除:
  ```bash
  git worktree remove worktrees/567-claude-skills
  ```
```

## Error Handling

このSkillはベストエフォート戦略を採用しています：

1. **Issue不明時**:
   - `gh issue list` を表示
   - ユーザーに正しいIssue番号を確認
   - 最近の10件のIssueをリスト表示

2. **Worktree既存時**:
   - 既存worktreeの状態を表示
   - 既存を使用するか確認
   - または上書き（`git worktree remove` 後に再作成）

3. **pre-commit失敗時**:
   - エラー詳細を表示
   - 修正が必要なファイルをリスト
   - 修正後に再実行を促す
   - 例: "ruff format failed on web/models.py"

4. **pre-push失敗時**:
   - テスト失敗詳細を表示
   - `build-and-test` Skill実行を提案
   - 失敗ログへのパスを表示

5. **PR作成失敗時**:
   - エラーメッセージを解析
   - 権限エラー → `gh auth refresh` を提案
   - ネットワークエラー → リトライを提案

## Best Practices

### プロジェクト固有の制約を徹底

1. **Issue番号必須**: ブランチ名・PR名に必ず含める
2. **mainブランチ直接作業禁止**: 必ずworktree使用
3. **PRマージ禁止**: `gh pr merge` は実行しない（ユーザーのみ）
4. **`--no-verify` 禁止**: フックは必ず実行

### 推奨ワークフロー

```bash
# 1. Issue作成（GitHub上またはgh CLI）
gh issue create --title "New Feature" --body "..."

# 2. このSkillでworktree作成
# "create worktree for #567"

# 3. 実装
cd worktrees/567-feature-name
# ... 実装作業 ...

# 4. コミット・PR作成（このSkillが自動化）
# git add, commit, push, gh pr create

# 5. レビュー・マージ（ユーザーが実行）
```

## Related Skills

- `build-and-test`: PR作成前に実行推奨
- `api-doc-sync`: FastAPI変更時は併用

## Implementation Notes

このSkillは以下の制約を自動的に適用します：

- **GitHub CLI必須**: `gh` コマンド使用
- **Worktree構造**: `/home/michihito/Working/totton_audio/worktrees/{issue}-{slug}`
- **ブランチ命名**: `feature/#{issue}-{slug}` または `fix/#{issue}-{slug}`
- **PR命名**: `#{issue} {title}` 形式
- **Co-Authored-By**: Claudeのクレジット自動追加

## Automation Level

**半自動実行**: Worktree作成と PR作成を自動化、実装はユーザーが行います。
