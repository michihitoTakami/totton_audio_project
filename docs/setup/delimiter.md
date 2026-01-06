# De-Limiter セットアップ & 運用手順 (#1102)

De-Limiter（AI Loudness Care）を **モデル取得 → 配置 → 設定 → ON** まで通せるようにするための手順をまとめます。高遅延ワーカの chunk/overlap 方式（4.0s / 0.25s）を前提とし、**推論サンプルレートは 44.1kHz 固定**です（#1096 方針）。

## 前提

- C++デーモンを ONNX Runtime 付きでビルドする
  - 例: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DDELIMITER_ENABLE_ORT=ON -DONNXRUNTIME_ROOT=/opt/onnxruntime`
  - `DELIMITER_ENABLE_ORT=OFF` のままではバックエンドが `Unsupported` を返し、常にバイパスになります。
- Python 依存をインストール（モデルDL/ONNXエクスポート用）
  - `uv sync --extra delimiter --extra onnxruntime`
  - GPU EP を使う場合は `--extra onnxruntime-gpu` に置き換え。

## 手順（モデル取得 → 配置 → 設定 → ON）

1. **モデル取得 + ハッシュ検証**
   ```bash
   uv run python scripts/delimiter/download_assets.py \
     --model jeonchangbin49-de-limiter \
     --sample-rate 44100 \
     --dest data/delimiter/weights
   ```
   - 取得元: upstream MIT (`https://github.com/jeonchangbin49/De-limiter`)
   - ハッシュ/メタデータ: `data/delimiter/weights/manifest.json`

2. **ONNX 生成**
   ```bash
   uv run python scripts/delimiter/export_onnx.py \
     --weights-dir data/delimiter/weights/jeonchangbin49-de-limiter/44100 \
     --output data/delimiter/weights/jeonchangbin49-de-limiter/44100/delimiter.onnx
   ```
   - 出力は git 非管理。`sha256sum delimiter.onnx` を控えておくと安全。

3. **設定反映（例: `config.json`）**
   - サンプル: `config_samples/delimiter/delimiter_44k.json`（44.1k系入力）、`config_samples/delimiter/delimiter_48k_resample.json`（48k入力を 44.1k 推論に正規化）
   - 主要項目
     - `delimiter.enabled`: `true` で高遅延パスを有効化（`false` のままだと Web UI で backend unavailable）
     - `delimiter.backend`: `ort` を指定（`bypass` はNo-op）
     - `delimiter.expectedSampleRate`: **モデルが期待するSR**。現行モデルは `44100` 固定（44.1k/48k入力どちらもこの値のまま運用）。許可値は `44100` または `48000` のみ。
    - `delimiter.chunkSec` / `overlapSec`: 推奨 `4.0` / `0.25`（#1232 デフォルト短縮）。`chunkSec > overlapSec` を守る。
     - `delimiter.ort.modelPath`: 上記で生成した ONNX へのパス（存在必須）。
     - `delimiter.ort.provider`: `cpu` / `cuda` / `tensorrt`（ONNX Runtime ビルドに含まれているもののみ使用可）。
     - `delimiter.ort.intraOpThreads`: ORTのスレッド数。`0` でデフォルト。
   - 反映後、デーモンを再起動（例: `./scripts/daemon.sh restart`）。

4. **ON にする**
   - Web UI: `http://127.0.0.1:11881/delimiter` のトグルを ON。`backendAvailable` が `true` か、`detail`/`fallback_reason` にエラーが出ていないかを確認。
   - API 直接操作: `GET /delimiter/status`, `POST /delimiter/enable`, `POST /delimiter/disable`。

## 44.1k / 48k の扱い

- **推論 SR は 44.1kHz 固定**。`expectedSampleRate` も 44100 に合わせる。
- 44.1k 入力: リサンプルなしでそのまま chunk/overlap へ。
- 48k 入力: 高遅延パス内で **48k → 44.1k リサンプル → 推論 → 44.1k → 48k リサンプル戻し**。現在は線形補間リサンプラを使用（品質と負荷のトレードオフに注意）。
- 48k ネイティブモデルを使う場合のみ `expectedSampleRate=48000` を検討する。それ以外で 48000 を設定するとモデルと不整合で失敗する。

## トラブルシュート

- **backend unavailable と表示される**
  - デーモン未起動 / `delimiter.enabled=false` / CMakeで `DELIMITER_ENABLE_ORT=OFF` のまま → デーモンを再起動し、設定とビルドフラグを確認。
- **モデルパス/プロバイダ関連のエラー**
  - `delimiter.ort.modelPath is empty / does not exist` → ONNX の配置とパスを再確認。
  - `Execution provider ... is not available` → ORT ビルドに含まれるプロバイダを選ぶ（`provider` を `cpu` に戻すか、対応プロバイダ付きで ORT を再ビルド）。
  - `ONNX Runtime backend is not enabled at build time` → `-DDELIMITER_ENABLE_ORT=ON` で再ビルド。
- **SR ミスマッチ/リサンプル失敗**
  - `delimiter.expectedSampleRate` が 44100/48000 以外になっていないか確認。
  - `delimiter input/output resample failed` が出る場合は入力SR設定や chunk/overlap を見直す。
- 詳細はデーモンログ（`/var/log/gpu_upsampler/daemon.log` など）と Web UI の `detail` フィールドで確認できる。

## ライセンスと配布物

- upstream MIT ライセンス: `data/delimiter/weights/LICENSE.upstream`
- 配布メタ情報: `data/delimiter/weights/manifest.json`
- プロジェクト全体の通知: `NOTICE.md`、`THIRD_PARTY_LICENSES.md` に De-limiter を明記済み
- 重み/ONNXは git 非管理。必要に応じて再ダウンロード/再生成して運用する。
