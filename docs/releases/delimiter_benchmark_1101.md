## #1101 De-limiter ベンチ/回帰テスト結果

- ベースコミット: a68f289 (origin/main)
- モデル: `data/delimiter/weights/jeonchangbin49-de-limiter/44100/delimiter.onnx`
- パラメータ: `chunk_sec=6.0`, `overlap_sec=0.25`, `target_sr=44100`, provider=`cpu`
- 入力: 自前生成の 12 秒ステレオサイン波（0.2 振幅）
- コマンド:
  - `uv run python scripts/delimiter/benchmark_streaming.py --input test_output/bench_input_44k.wav --model .../delimiter.onnx --provider cpu --measure-resources --report test_output/reports/bench_44k.json`
  - `uv run python scripts/delimiter/benchmark_streaming.py --input test_output/bench_input_48k.wav --model .../delimiter.onnx --provider cpu --measure-resources --report test_output/reports/bench_48k.json`

### 計測サマリ

| 入力SR | mean ms/chunk | p95 ms/chunk | throughput_x | RTF | CPU avg/max | GPU avg/max | 備考 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 44.1k → 44.1k | 287.3 | 318.4 | 13.92x | 0.072 | 576.6% / 668.2% | 27.0% / 29.0% | chunk=6s, hop=5.75s |
| 48k → 44.1k → 48k | 338.5 | 408.0 | 11.82x | 0.085 | 635.5% / 665.7% | 26.0% / 31.0% | 48k入力をターゲットSRに正規化 |

- どちらも `error_rate=0.0`, `failed_chunks=[]`（新しい fallback 計測項目）。
- 初期遅延は chunkSec に一致（6.0s）。重複領域 hop=5.75s。

### 回帰テスト拡充

- `scripts/delimiter/benchmark_streaming.py`
  - `--target-sr` でモデル側SRを明示、`--fallback-on-error` で推論例外時に当該chunkのみバイパスし `error_rate`/`failed_chunks` を記録。
  - ラウンドトリップ実行ヘルパー `run_streaming_benchmark_roundtrip` を追加（入力SRへ戻した状態で評価可能）。
- `tests/python/test_delimiter_benchmark.py`
  - 48k入力→44.1k推論→48k戻しの往復が長さを保持することを検証。
  - 1chunk失敗時にバイパスしつつ継続する `fallback_on_error` の回帰テストを追加。

### メモ

- `uv sync --group dev --extra delimiter --extra onnxruntime --extra benchmark` で依存を揃える（onnxruntime/onnx/psutil/torch 等）。
- 計測用WAV/JSONは `test_output/` に生成（git管理外）。再取得可能なのでクリーンアップしても問題なし。
