# Build and Test

C++/CUDA実装後の完全ビルド・テスト・レポート生成を自動化します。

## Description

このSkillは、プロジェクトの完全ビルドと全テストスイートの実行を自動化し、失敗したテストの詳細分析と推奨アクションを提供します。

## Trigger Words

- `build and test`
- `run all tests`
- `full build`
- `ビルドしてテスト`

## Requirements

- CMake がインストールされていること
- CUDA Toolkit（GPU テスト用）
- 実行ディレクトリ: プロジェクトルート (`/home/michihito/Working/totton_audio/`)
- 十分なディスクスペース（build/ディレクトリ）

## Execution Steps

```bash
# 1. ビルドディレクトリの確認・作成
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. インクリメンタルビルド（並列）
cmake --build build -j$(nproc)

# 3. テスト実行（全17ターゲット）
cd build && ctest --output-on-failure

# 4. 失敗したテスト詳細収集（必要に応じて）
# 各失敗ターゲットを個別実行し、完全ログ取得
```

## Parameters

なし（自動的に全ターゲットをビルド・テスト）

## Expected Output

### 成功時:
```markdown
# ビルド・テスト結果

## ビルド: ✅ 成功 (3分42秒)
- 変更ファイル: 12個
- 再コンパイル: src/alsa_daemon.cpp, src/eq_parser.cpp

## テスト: ✅ 全て成功 (17/17実行, 17成功, 0失敗)

✨ 全てのテストが正常に完了しました！
```

### 失敗時:
```markdown
# ビルド・テスト結果

## ビルド: ✅ 成功 (3分42秒)

## テスト: ⚠️ 一部失敗 (17/17実行, 15成功, 2失敗)

### 失敗したテスト
1. **cpu_tests.EQParserTest.ParseInvalidFile**
   - エラー: Expected exception not thrown
   - ファイル: tests/cpp/test_eq_parser.cpp:45
   - 原因: 例外ハンドリングが不足

2. **zmq_tests.ZeroMQInterfaceTest.TimeoutHandling**
   - エラー: Timeout expected but got immediate response
   - ファイル: tests/cpp/test_zeromq_interface.cpp:89
   - 原因: タイムアウトロジックが正しく動作していない

### 推奨アクション
- [ ] EQParserの例外ハンドリングを確認（test_eq_parser.cpp:45）
- [ ] ZeroMQタイムアウトロジックをレビュー（src/zeromq_interface.cpp）
- [ ] 該当テストを個別実行して詳細を確認
```

## Error Handling

このSkillはベストエフォート戦略を採用しています：

1. **ビルド失敗時**:
   - コンパイルエラー箇所を特定
   - 該当ファイルと行番号をレポート
   - エラーメッセージを詳細表示

2. **テスト失敗時**:
   - 失敗したテストのみ詳細表示
   - 成功したテストは簡潔に報告
   - 失敗原因の推測と修正提案

3. **GPU不足エラー時**:
   - GPU依存テストをスキップ
   - CPU-onlyテストは継続実行
   - GPUリソース不足を警告

4. **タイムアウト時**:
   - 長時間実行テストを警告
   - 5分以上のテストは中断
   - 残りのテストは継続

## Best Practices

- **pre-push前の実行を推奨**: コミット前に必ず実行
- **worktree内で実行**: mainブランチでは実行しない
- **並列ビルド**: `-j$(nproc)` で最大並列度を使用
- **クリーンビルドが必要な場合**: `rm -rf build/` してから実行

## Related Skills

- `worktree-pr-workflow`: PR作成前にこのSkillを実行
- `api-doc-sync`: Web API変更時は両方実行を推奨

## Implementation Notes

このSkillは以下の既存ツールを活用します：

- `CMakeLists.txt`: ビルドターゲット定義
- `ctest`: テスト実行フレームワーク
- `scripts/deployment/run_tests.sh`: 差分ベーステスト（pre-push hook）
- GoogleTest: C++テストフレームワーク

## Automation Level

**完全自動実行**: ユーザー入力不要で全プロセスを実行します。
