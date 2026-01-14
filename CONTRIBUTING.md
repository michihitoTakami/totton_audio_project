# Contributing Guide / コントリビューションガイド

このリポジトリへのコントリビューションありがとうございます！
初見の方が「ローカルでビルド→テスト→PR」まで到達できるように、手順とルールをまとめます。

## 1. まず読むもの

- `README.md`（全体像）
- `INSTALL.md`（依存関係〜ビルド〜テストの手順）
- `docs/`（用途別ドキュメント）
- 公開前チェック: `docs/repo_hygiene.md`

## 2. 開発環境セットアップ（最短）

基本は `INSTALL.md` に沿ってください。特にこの順序が安全です：

```bash
# Python 依存
uv sync --all-groups

# C++/CUDA ビルド（compile_commands.json も生成）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j"$(nproc)"
```

## 3. テスト / 品質ゲート

PR前に最低限ここまで通してください（詳細は `INSTALL.md` 参照）：

```bash
# C++ tests
./build/cpu_tests
./build/gpu_tests

# Python tests
uv run pytest

# pre-commit（pre-push 相当）
pre-commit run --hook-stage pre-push
```

## 4. Issue / PR の流れ

### 4.1 Issue

- 既存Issueを検索し、なければ新規作成してください。
- 再現手順、期待結果、実際の結果、環境情報（OS/GPU/Driver/CUDAなど）を書くと助かります。

### 4.2 ブランチ/PR 命名（必須）

Issue番号を必ず含めてください。

- ブランチ名: `feature/123-topic` / `fix/456-bug`
- PRタイトル: `123 変更の説明` / `Fix 456: バグの説明`

## 5. ライセンス / 権利（DCO）

このプロジェクトでは **DCO (Developer Certificate of Origin) 1.1** による同意を採用します。
すべてのコミットに `Signed-off-by` を付けてください（`git commit -s`）。

```bash
git commit -s
```

### DCO 1.1（要旨）

あなたは提出する変更について、以下のいずれかを満たすことを表明します：

1) 変更はあなたが作成し、当該プロジェクトのライセンスの下で提供する権利がある
2) 変更は適切なライセンスの下で入手し、そのライセンス条件の下で提出できる
3) 変更は (1) または (2) を満たす人物から提供され、あなたがそれを提出する権利を持つ
4) 変更が公開されること、氏名とメールアドレスが記録されることに同意する

全文: https://developercertificate.org/

## 6. English (short)

Please read `README.md` and `INSTALL.md` first.
Before opening a PR, run tests and `pre-commit run --hook-stage pre-push`.
We use **DCO 1.1**: sign every commit with `git commit -s`.
