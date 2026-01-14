# リポジトリ衛生チェック（公開前）

公開前に「混入しがちなもの（生成物、秘密情報、ネストgit等）」を排除・予防するための手順です。

## 方針

- **ネストgit（例: `external/**/.git`）は禁止**（意図せず履歴/機密が混入するリスクが高い）
- **submodule は使わない**（`.gitmodules` を置かない）
- 生成物（`build/`、キャッシュ、仮想環境など）は **追跡しない**
- シークレット検知は **pre-commit の `detect-secrets`** を基準に運用する

## すぐ実行できるチェック

### 1) 追跡ファイルだけで確認（CIでも有効）

```bash
./scripts/repo/check_repo_hygiene.sh
```

### 2) コミット直前（staged）で確認

```bash
./scripts/repo/check_repo_hygiene.sh --staged
```

## シークレットスキャン

このリポジトリは `detect-secrets` を `pre-commit` で有効化しています（`.secrets.baseline`）。

```bash
uv run pre-commit run detect-secrets --all-files
```

ベースライン更新が必要な場合:

```bash
uv run detect-secrets scan --baseline .secrets.baseline
```

## 生成物が追跡されてしまった場合の対処

```bash
git rm -r --cached <path>
```

そのうえで `.gitignore` にパターンを追加し、再発防止してください。

## worktree について

開発中に `git worktree` を使う場合は `worktrees/` 配下に作成し、このディレクトリは追跡しません（`.gitignore`）。

