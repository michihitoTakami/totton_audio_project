# テスト実行手順

Magic Box Projectのテスト実行方法。

## テスト概要

テストはGoogleTestフレームワークを使用しています。

### テストカテゴリ

| カテゴリ | 説明 | GPU必要 |
|---------|------|---------|
| CPUテスト | 基本機能、ユーティリティ | No |
| GPUテスト | GPU畳み込み処理 | Yes |
| 統合テスト | ZeroMQ、Auto-Negotiation等 | No |

## テストの実行

### 全CPUテスト（GPU不要）

```bash
./build/cpu_tests
./build/zmq_tests
./build/auto_negotiation_tests
./build/soft_mute_tests
./build/base64_tests
./build/error_codes_tests
./build/fallback_manager_tests
```

### 全GPUテスト（GPU必要）

```bash
./build/gpu_tests
./build/crossfeed_tests
```

### 一括実行スクリプト

```bash
./scripts/deployment/run_tests.sh
```

## 個別テストの詳細

### cpu_tests

基本的なCPU処理のテスト。

```bash
./build/cpu_tests

# 特定のテストのみ実行
./build/cpu_tests --gtest_filter="*RingBuffer*"
```

含まれるテスト:
- リングバッファ操作
- オーディオI/O
- 設定ファイルパース
- ユーティリティ関数

### gpu_tests

GPU畳み込み処理のテスト。

```bash
./build/gpu_tests
```

含まれるテスト:
- フィルタ係数のロード
- FFT畳み込み処理
- Overlap-Save アルゴリズム
- マルチレート対応

### zmq_tests

ZeroMQ通信のテスト。

```bash
./build/zmq_tests
```

含まれるテスト:
- REQ/REPパターン
- PUB/SUBパターン
- メッセージシリアライズ
- コマンド送受信

### auto_negotiation_tests

Auto-Negotiation（DAC性能検出・レート最適化）のテスト。

```bash
./build/auto_negotiation_tests
```

含まれるテスト:
- DAC性能検出ロジック
- 最適倍率の算出
- レートファミリー判定（44.1k系/48k系）

### soft_mute_tests

Soft Mute（フェード制御）のテスト。

```bash
./build/soft_mute_tests
```

含まれるテスト:
- フェードイン/フェードアウト
- 同族内レート切り替え
- クロス族レート切り替え

### crossfeed_tests

Crossfeed（HRTF）のテスト。

```bash
./build/crossfeed_tests
```

含まれるテスト:
- HRTF係数のロード
- Crossfeed処理
- 頭サイズ切り替え
- ダブルバッファリング

### base64_tests

Base64エンコード/デコードのテスト。

```bash
./build/base64_tests
```

### error_codes_tests

エラーコード定義のテスト。

```bash
./build/error_codes_tests
```

### fallback_manager_tests

Fallback Manager（GPU負荷監視・自動縮退）のテスト。

```bash
./build/fallback_manager_tests
```

## GoogleTestオプション

### 特定のテストのみ実行

```bash
# パターンマッチ
./build/cpu_tests --gtest_filter="*RingBuffer*"

# 複数パターン
./build/cpu_tests --gtest_filter="*Ring*:*Buffer*"

# 除外
./build/cpu_tests --gtest_filter="-*Slow*"
```

### テスト一覧の表示

```bash
./build/cpu_tests --gtest_list_tests
```

### 詳細出力

```bash
./build/cpu_tests --gtest_print_time=1
```

### 繰り返し実行

```bash
./build/cpu_tests --gtest_repeat=10
```

### 失敗時に停止

```bash
./build/cpu_tests --gtest_break_on_failure
```

## Pythonテスト

Python側のテストはpytestを使用します。

```bash
uv run pytest tests/ -v
```

### カバレッジ付き

```bash
uv run pytest tests/ --cov=web --cov-report=term-missing
```

## CI/CD

GitHub Actionsでは以下のテストが自動実行されます：

1. CPUテスト（全プラットフォーム）
2. GPUテスト（GPU環境のみ）
3. Pythonテスト
4. リンター（clang-tidy、ruff）

## トラブルシューティング

### GPU not available

```
CUDA error: no CUDA-capable device is detected
```

GPUが利用できない環境です。CPUテストのみ実行してください。

### フィルタファイルが見つからない

```
Failed to load filter coefficients
```

フィルタ係数を生成してください：

```bash
uv run python scripts/filters/generate_minimum_phase.py --generate-all
uv run python scripts/filters/generate_linear_phase.py --generate-all
```

### セグメンテーションフォルト

デバッグビルドで原因を特定：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# gdbで実行
gdb ./build/gpu_tests
(gdb) run
(gdb) bt   # バックトレース
```

### テストがタイムアウト

長時間実行テストにはタイムアウトを延長：

```bash
./build/gpu_tests --gtest_timeout=300000  # 5分
```
