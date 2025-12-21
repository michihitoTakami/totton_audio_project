## De-limiter ストリーミング分割設計（chunk/overlap/crossfade） (Fix #1009)

### 目的
- **6〜10秒遅延を前提**に、De-limiter のような「固定長/高コスト推論」を **連続再生**できる形にする。
- chunk境界での **クリック/境界アーティファクトを抑制**するため、**overlap + crossfade（OLA）** を採用する。
- 次の実装タスク（#1010: 高遅延処理パス、#1014: 安全設計、#1017: 推論バックエンド抽象化）と **矛盾しない前提**を固定する。
- 併せて、モデル選定と 44.1k/48k の SR 方針は **別ドキュメント**で確定させる: `docs/specifications/delimiter_model_sr_strategy_1096.md`（Issue #1096）

---

### 非目的（このIssueではやらない）
- RTキャプチャスレッドから推論を呼ぶ実装（→ #1010）
- 失敗時の最終的な安全策（閾値/自動OFF/メトリクス/復帰）（→ #1014）
- ONNX Runtime / TensorRT の本体実装（→ #1017 / follow-up）

---

## 1. 方式概要

### 入力/出力の前提
- **挿入位置**: アップサンプリング前（入力 44.1kHz/48kHz）
- **処理単位**: 固定長 chunk（例: 6.0秒）を推論へ渡し、復元後 chunk を受け取る
- **ステレオ同期**: L/R は **同一境界**・同一窓・同一chunk長で扱う（片ch単独で境界処理しない）

### アルゴリズム: Overlap-Add + Crossfade
- chunk長: \(N\) frames
- overlap長: \(O\) frames
- hop長: \(H = N - O\) frames
- chunk i の出力を時刻 \(iH\) に配置し、overlap領域は **窓で加重平均**して合成する

---

## 2. パラメータ（デフォルト案）

### デフォルト（案）
- **chunkSec**: `6.0` 秒
- **overlapSec**: `0.25` 秒（250ms）
- **window**: raised-cosine（Hannの半周期相当）

### フレーム換算
- sampleRate を \(Fs\) とすると
  - `chunkFrames = round(chunkSec * Fs)`
  - `overlapFrames = round(overlapSec * Fs)`
  - `hopFrames = chunkFrames - overlapFrames`

### 制約
- **必須**: `chunkFrames > overlapFrames`（hopが正になる）
- **推奨**: `overlapFrames <= chunkFrames / 2`
  - 1つのchunkの先頭フェード領域と末尾フェード領域が干渉しないため

---

## 3. クロスフェード窓

### raised-cosine（推奨）
fade-in（0→1）:
\[
w(t) = 0.5 - 0.5\cos(\pi t),\quad t \in [0,1]
\]

離散化（overlapFrames = O）:
- `t = i/(O-1)` for i=0..O-1
- `fadeIn[i] = 0.5 - 0.5*cos(pi*t)`
- `fadeOut[i] = fadeIn[O-1-i]`

合成（overlap領域）:
- 直前chunk: `fadeOut`
- 次chunk: `fadeIn`
- **同一 i で `fadeIn[i] + fadeOut[i] = 1`** になるため、理想的には正規化無しでも振幅が保たれる
  - 実装では数値誤差対策として `wsum` で正規化して良い

---

## 4. 起動直後 / ON切替 / 設定変更

### ON切替時（遅延を許容するモードに入る）
- **オフ時は遅延ゼロ**（従来パス）
- ONにした瞬間から:
  - まず **入力をバッファ**し、`chunkFrames` 溜まるまで推論へ渡せない
  - 出力は **無音**を出してタイムラインを維持（= 再生は続くが無音）
  - 最初の推論chunkが戻ったら、**フェードイン**して復元音へ移行

### 設定変更（chunk/overlap/window/推論backend切替等）
- クリック防止のため、**SoftMuteフェードアウト→状態リセット→フェードイン** を基本手順とする
- バッファ/overlap状態は必ずクリアし、境界不整合を残さない

---

## 5. 失敗時の挙動（方針）

### 目標
- 推論失敗/過負荷時でも **クリック/爆音/タイムライン崩壊を避ける**

### 最低限の方針（#1014で詳細化）
- あるchunkの推論が失敗した場合:
  - **そのchunkは bypass（入力そのまま）にフォールバック**
  - 前後chunkとの接続は同じ crossfade 窓で行う（境界の急変を避ける）
- 連続失敗時:
  - 自動で delimiter を OFF（bypass固定）し、ユーザー操作で再ON可能

---

## 6. 参照実装（このリポジトリ内）

### C++（設計検証用）
- `include/audio/overlap_add.h`
  - `AudioUtils::makeRaisedCosineFade()`
  - `AudioUtils::overlapAddStereoChunks()`

この実装は **設計の数式/窓の一致**を確認する目的の参照であり、RT経路へ直接組み込むのは #1010 で行う。

---

## 7. 性能計測とデフォルト決定（Fix #1015）

- ストリーミング時の **平均処理速度>1.0x**（throughput_x >= 1）が目標。chunkSec=6.0 / overlapSec=0.25 を基本とし、Jetson Orin Nano（最小ターゲット）で計測して維持できるか確認する。
- 計測スクリプト: `scripts/delimiter/benchmark_streaming.py`
  - 例:
    `uv sync --extra onnxruntime --extra benchmark`
    `uv run python scripts/delimiter/benchmark_streaming.py --input test_data/example.wav --model /path/to/delimiter.onnx --provider cpu --chunk-sec 6.0 --overlap-sec 0.25 --measure-resources --report reports/delimiter_bench.json`
  - 出力: chunkごとの推論時間、平均/95パーセンタイル、throughput_x（速いほど良い）、推定初期遅延（chunkSec）、hop秒（chunkSec - overlapSec）、CPU/GPU使用率（psutil/nvidia-ml-py3がある場合）。
- デフォルト運用: throughput_x が 1 未満になった場合は
  - chunkSec を短くする（初期遅延は増えるが計算量が減る）
  - overlapSec を縮める（境界品質とトレードオフ）
  - 推論プロバイダを GPU/TensorRT に切替える
