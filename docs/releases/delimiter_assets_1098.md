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

- upstream ハッシュ（`tr -d ':'` で生ハッシュ）
  - `all.pth` SHA256 `df:d9:1f:96:05:c6:55:38:ac:ab:eb:9d:56:50:c2:11:19:61:1a:9e:04:93:de:3e:1f:30:05:76:a9:92:5e:c9` (9 424 379 bytes)
  - `all.json` SHA256 `b5:17:a0:19:1f:44:ed:88:9c:83:9d:79:df:f6:e0:07:5d:c5:40:57:d1:ab:0f:1d:55:58:1a:95:bb:90:ae:27` (25 455 bytes)
- `manifest`: `data/delimiter/weights/manifest.json`

### 注意点
- 推論 SR は **常に 44100**。48k 系は #1100 のリサンプル統合後に有効化する。
- ONNX Runtime ビルドに応じて `provider` を `cpu/cuda/tensorrt` から選択。サンプルは安全のため `cpu` をデフォルトにしている。
