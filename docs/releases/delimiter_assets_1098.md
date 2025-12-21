## #1095-3 De-Limiter: モデル配備と設定サンプル整備（#1098）

### 背景
- 採用モデル: `jeonchangbin49/De-limiter`（MIT、推論 SR=44100）[#1096]
- 配布形態・命名規約・設定サンプルを整備し、Web/UI で有効化できる状態を作る。

### 決定事項
- **配置ルール**: `data/delimiter/weights/<model_id>/<sr>/` に重み・ONNX を置く。現在は `jeonchangbin49-de-limiter/44100/` のみ。
- **配布方針**: バイナリは git 非管理。`scripts/delimiter/download_assets.py` で取得し、SHA256 で検証する。
- **ONNX**: `scripts/delimiter/export_onnx.py` で `delimiter.onnx` をローカル生成（入力 `(1,2,T)` float32、動的 T）。git 非管理。
- **設定サンプル**: 2 つの JSON を `config_samples/delimiter/` に追加。
  - `delimiter_44k.json`: 入力 44.1k 前提。推論 SR=44100。
  - `delimiter_48k_resample.json`: 入力 48k。推論 SR=44100 なので 48k→44.1k→48k のリサンプル実装 (#1100) が入るまで実運用ではバイパス動作。
- **ライセンス/NOTICE**: upstream MIT を `data/delimiter/weights/LICENSE.upstream` に同梱し、`NOTICE.md` / `THIRD_PARTY_LICENSES.md` に追記。

### 取得・検証手順（44100）

```bash
uv run python scripts/delimiter/download_assets.py \
  --model jeonchangbin49-de-limiter \
  --sample-rate 44100 \
  --dest data/delimiter/weights

uv run python scripts/delimiter/export_onnx.py \
  --weights-dir data/delimiter/weights/jeonchangbin49-de-limiter/44100 \
  --output data/delimiter/weights/jeonchangbin49-de-limiter/44100/delimiter.onnx
```

- upstream ハッシュ
  - `all.pth` SHA256 `dfd91f9605c65538acabeb9d5650c21119611a9e0493de3e1f300576a9925ec9` (9 424 379 bytes)
  - `all.json` SHA256 `b517a0191f44ed889c839d79dff6e0075dc54057d1ab0f1d55581a95bb90ae27` (25 455 bytes)
- `manifest`: `data/delimiter/weights/manifest.json`

### 注意点
- 推論 SR は **常に 44100**。48k 系は #1100 のリサンプル統合後に有効化する。
- ONNX Runtime ビルドに応じて `provider` を `cpu/cuda/tensorrt` から選択。サンプルは安全のため `cpu` をデフォルトにしている。

---

## オフライン検証パス（#1099 実行例）

### PyTorch weights 版（WAV→WAV）

```bash
uv run python scripts/delimiter/offline_wav_to_wav.py \
  --input test_data/input.wav \
  --output /tmp/output.wav \
  --backend delimiter \
  --expected-sample-rate 44100 \
  --chunk-sec 6.0 \
  --overlap-sec 0.25 \
  --resample-back \
  --debug-dir /tmp/delimiter_debug \
  --report /tmp/delimiter_report.json
```

- 入出力ともに **peak / clip率 / RMS / LUFS** を標準出力で確認できる（LUFSは `pyloudnorm` が無い場合 `n/a`）。
- `--expected-sample-rate` で推論SRを指定（既定 44100）。入力が異なる場合は自動リサンプルし、`--resample-back` で元SRへ戻す。
- `--report` には上記メトリクスと処理時間/RTF、デバッグパスをJSONで保存。

### ONNX Runtime 版（WAV→WAV）

```bash
uv run python scripts/delimiter/onnx_wav_to_wav.py \
  --input test_data/input.wav \
  --output /tmp/output.wav \
  --model data/delimiter/weights/jeonchangbin49-de-limiter/44100/delimiter.onnx \
  --provider cpu \
  --expected-sample-rate 44100 \
  --chunk-sec 6.0 \
  --overlap-sec 0.25 \
  --resample-back \
  --report /tmp/delimiter_report_ort.json
```

- 出力はPyTorch版と同様に peak / clip率 / RMS / LUFS / RTF を表示。
- `provider` で `cpu/cuda/tensorrt` を切替可能（環境に応じて ORT のビルドが必要）。
