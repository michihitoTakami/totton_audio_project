## De-Limiter: モデル選定 & 44.1k/48k サンプルレート運用方針（Issue #1096）

## 背景
EPIC #1095 の De-Limiter は、現状 C++ 側の `InferenceBackend` がプレースホルダ（ORT 未リンク）で、Web から ON にしても常に bypass になる。
実運用では 44.1kHz 系と 48kHz 系の両方を扱う必要があるため、**商用利用可能なモデル候補**と **SR(サンプルレート) 戦略**を先に確定する。

## 結論（決定事項）
- **採用モデル**: `jeonchangbin49/De-limiter` の学習済みモデル（MIT License）
- **SR 戦略**: **(A) 44.1kHz を推論の正規 SR とし、48kHz 入力は高品質リサンプルで対応**
  - 入力が 48kHz のとき: `48k -> 44.1k (推論) -> 48k`（推論は常に 44.1k）
  - 入力が 44.1kHz のとき: リサンプルなし

この決定は、#1097/#1098/#1099/#1100 の設計前提になります。

---

## 候補モデル比較（2件）

| 候補 | 形態 | ライセンス | 推論SR | 推論方式 | 採用判断 |
|---|---|---|---:|---|---|
| De-limiter (upstream weights) | `all.pth` + `all.json` | MIT | 44100 | PyTorch（Asteroid系） | **採用（モデルのソース・オブ・トゥルース）** |
| De-limiter (ONNX export) | `delimiter.onnx` | MIT（同一ソース由来） | 44100 | ONNX Runtime（CPU/CUDA/TensorRT） | **採用（C++ 統合の主対象）** |

### 採用理由（De-limiter）
- upstream が **MIT** で、学習済み重みが `weight/` に同梱されている（別配布手続き不要）
- リポジトリ内に既に PoC（オフライン変換、ONNX/ORT 実行）が存在し、#1095 の「高遅延前提」と整合

---

## ライセンス根拠（一次情報）
- upstream リポジトリ: `https://github.com/jeonchangbin49/De-limiter`
- LICENSE（MIT）: `https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/LICENSE`
- 重み同梱（README 記載 / weight ディレクトリ）:
  - README: `https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/README.md`
  - weight: `https://github.com/jeonchangbin49/De-limiter/tree/main/weight`

> 注意: 「重みファイル単体の別ライセンス」表記は upstream 側で見当たらないため、当面は **リポジトリライセンス（MIT）に包含される前提**で扱う。#1098 で再確認し、必要なら NOTICE/ATTRIBUTION を強化する。

---

## SR 戦略の詳細（A: 44.1k正規化 + 48kリサンプル）

### 方針
- **推論 SR は常に 44100Hz**
- **48kHz 系統入力**は推論の前後でリサンプルして整合させる

### 理由
- 現状の PoC/設定が 44.1kHz 前提（例: `scripts/delimiter/offline_wav_to_wav.py` / `onnx_wav_to_wav.py` は 44.1k に正規化）
- 48k ネイティブ学習済みモデルの「商用可 + 公開済み」が短時間調査で確認できていない
- De-Limiter は高遅延（6〜10s）前提のため、リサンプルの追加コストが全体に対して相対的に小さい

### 実装上の位置（#1100 向け）
- delimiter の高遅延ワーカ内で
  - 入力バッファを **44.1k に変換**
  - **44.1k 側で chunk/overlap/crossfade** を行い推論
  - 出力を **元 SR（48k）へ戻す**
- 量産（C++）側のリサンプラは既存方針通り **libsoxr 等の高品質**を使用する（PoC の `scipy.signal.resample_poly` は検証用）

---

## 次の Issue への引き継ぎ（設計図）

### #1097（C++ ORT推論バックエンド）
- `delimiter.expectedSampleRate` は **「モデルが期待する SR」= 44100** を意味するものとして扱う
- 入力 SR 不一致（例: 48000）は **backend で Unsupported にしない**（※ safety上の最終判断は #1100）
  - 代わりに、backend は「推論SR=44100」を前提に入出力 shape を厳密に扱う
- ONNX の I/O 仕様（例: 入力 `(1,2,T)`）は C++ 実装で明確に固定・検証する

### #1098（モデル配備/設定/ライセンス表記）
- upstream MIT を根拠に **NOTICE/ATTRIBUTION** を整備済み（`data/delimiter/weights/LICENSE.upstream` / `NOTICE.md` / `THIRD_PARTY_LICENSES.md`）
- 重み/ONNX は **外部DL + ローカル生成** 方針
  - 取得: `scripts/delimiter/download_assets.py`（SHA256検証、manifest: `data/delimiter/weights/manifest.json`）
  - ONNX: `scripts/delimiter/export_onnx.py` で `data/delimiter/weights/jeonchangbin49-de-limiter/44100/delimiter.onnx` を生成（git非管理）
- 設定サンプル（SR=44100/48k系）: `config_samples/delimiter/*.json` を参照

### #1099（オフライン検証スクリプト）
- 現状の「44.1k 正規化 + resample-back」設計はこの SR 方針と一致
- メトリクス（peak/LUFS/RTF）を追加する場合も、**推論SRは44100固定**を前提にする

### #1100（リアルタイム統合/フェイルセーフ）
- 48k 入力時は **推論前後のリサンプル**が必要（bypass ではなく「処理できる」状態へ）
- SafetyController の reason は、少なくとも
  - model missing / backend init fail
  - SR mismatch（= resample 未実装/失敗）
  - overload
  を区別して UI/API に伝播できるようにする
