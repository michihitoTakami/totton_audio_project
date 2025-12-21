## De-limiter モデル資産配置ガイド (#1098)

De-limiter の重み・ONNX を配置するための規約をまとめます。巨大バイナリは **git で管理しません**。必要に応じてローカルに取得し、ハッシュで検証してください。

### ディレクトリ規約

```
data/delimiter/weights/
└── <model_id>/<sample_rate>/
    ├── all.pth          # PyTorch state_dict（upstreamそのまま）
    ├── all.json         # upstream の学習設定
    ├── delimiter.onnx   # all.pth からローカルでエクスポート（未追跡）
    └── LICENSE.upstream # upstream MIT（モデル単位で配置）
```

- 現行モデル: `model_id=jeonchangbin49-de-limiter`, `sample_rate=44100`
- 48k 系統は #1100 で実装予定のリサンプル経路が前提。**推論 SR は 44100 固定**（#1096 決定）。

### 配布方針

- バイナリは **git 非管理**。`.gitignore` で `weights/**` を除外し、メタ情報のみ管理。
- 取得・検証は `scripts/delimiter/download_assets.py` を利用。
- ONNX は `scripts/delimiter/export_onnx.py` でローカル生成。

### 取得手順（44100 / jeonchangbin49-de-limiter）

```bash
uv run python scripts/delimiter/download_assets.py \
  --model jeonchangbin49-de-limiter \
  --sample-rate 44100 \
  --dest data/delimiter/weights
```

- 取得元
  - `all.pth`: https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/weight/all.pth
    SHA256=`df:d9:1f:96:05:c6:55:38:ac:ab:eb:9d:56:50:c2:11:19:61:1a:9e:04:93:de:3e:1f:30:05:76:a9:92:5e:c9`（9 424 379 bytes、`tr -d ':'` で生ハッシュ）
  - `all.json`: https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/weight/all.json
    SHA256=`b5:17:a0:19:1f:44:ed:88:9c:83:9d:79:df:f6:e0:07:5d:c5:40:57:d1:ab:0f:1d:55:58:1a:95:bb:90:ae:27`（25 455 bytes）
  - `LICENSE.upstream`: https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/LICENSE

### ONNX 生成

```bash
uv run python scripts/delimiter/export_onnx.py \
  --weights-dir data/delimiter/weights/jeonchangbin49-de-limiter/44100 \
  --output data/delimiter/weights/jeonchangbin49-de-limiter/44100/delimiter.onnx
```

- 出力された `delimiter.onnx` は git 非管理。`sha256sum delimiter.onnx` で検証値を記録してください。
- ONNX Runtime の I/O 形状: `(1, 2, T)` の float32 （チャネルファースト）。

### サンプル設定

- 44.1k 系: `config_samples/delimiter/delimiter_44k.json`
- 48k 入力（リサンプル前提）: `config_samples/delimiter/delimiter_48k_resample.json`

上記の `delimiter.ort.modelPath` は本ディレクトリの `delimiter.onnx` を参照します。48k 系は SR ミスマッチ対策のリサンプル実装が入るまでバイパス動作となります。
