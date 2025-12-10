# Float64 GPU Upsampler Option (RFC)

## 背景
- GPU アップサンプラ（Overlap-Save 畳み込み）は主経路が `float32`。  
- 最低 640k taps の FIR を 16x で畳み込むため FFT サイズは約 1,048,576 (block 8,192 想定)。
- 最低位相 EQ の再構成部だけは `cufftDouble` を使用するが、最終フィルタは `float` にダウンコンバート。
- ステレオでも double 精度で計算量に余裕がある試算があるため、64bit 経路のオプション化を検討する。

## 64bit 化の影響（概算）
- FFT バッファ (R2C/C2R, N≈1,048,576):  
  - real double ≈ 8.0 MB / buffer  
  - complex double (N/2+1) ≈ 8.4 MB / buffer  
  - 主要バッファ（入力, 畳み込み結果, フィルタ FFT A/B, original, conv work）だけで ~50–70 MB 想定（float32 の約 2×）。
- オーバーラップ／ストリーミング用バッファも 2×。Jetson Orin Nano ではピン止めホストメモリ枯渇リスクあり。
- cuFFT: `CUFFT_D2Z`/`CUFFT_Z2D` は Jetson でスループット低下が大きい可能性（SM 8.7 の FP64 比率が低い）。

## 必要なコード変更の概要
- データ型の抽象化: `using Sample = float` をヘッダで型エイリアス化し、`float` 直書きを除去（ホスト/デバイス/バッファ定義一式）。
- cuFFT プランとバッファ: `cufftHandle` は `CUFFT_D2Z`/`CUFFT_Z2D`、`cufftDoubleComplex` を選択できるようにする（EQ 部の double バッファは流用可）。
- CUDA カーネル: `zeroPadKernel`/`complexMultiplyKernel`/`scaleKernel` を double 対応版にする（テンプレート化または `#if`）。ストリーミング用コピー経路も double に合わせる。
- フィルタ係数ロード: 係数ファイルは現状 float32。ダブル化する場合は (a) 読み込み後に double へ拡張, (b) 新たに float64 係数を配布、のいずれかを選択。
- ストリーミング/パーティション畳み込み: オーバーラップバッファ、tail accumulator、history バッファ等を新しいスカラー型に合わせる。
- 出力型: DAC へ渡す前に `float` へダウンコンバートするか、終端まで double で保持するかをオプション化（互換性を優先するならダウンコンバート）。

## オプション化の案
- CMake オプション `GPU_UPSAMPLER_USE_FLOAT64` (デフォルト OFF)。  
- オン時に:
  - 型エイリアスを double に切替
  - cuFFT プランを double 用に組み立て
  - double 対応カーネルをビルド
  - 出力を float32 へダウンコンバートするステップを入れる（またはフラグで切替）

## 検証観点
- 速度: PC dGPU と Jetson Orin Nano でリアルタイム処理が維持できるか（16x/8x/4x）。GPU 利用率・レイテンシ計測。
- メモリ: VRAM とピン止めホストメモリの消費を計測し、既存デーモンの常駐メモリと合わせて余裕を確認。
- 音質/数値: float32 との RMS 差分、SNDR/THD などを比較。EQ 適用時の誤差も評価。
- 回帰: オプション OFF 時に既存の性能・音質が変わらないこと。

## ステップ案
1) 型エイリアスとカーネル/バッファのテンプレート化で float/double を切替可能にする。  
2) cuFFT プランを型に従って生成するユーティリティを追加。  
3) ビルドフラグ `GPU_UPSAMPLER_USE_FLOAT64` を追加し、デフォルト OFF。  
4) ベンチ/検証スクリプトで PC/Jetson の計測を取り、結果を docs に追記。  

## リスク
- Jetson では FP64 帯域不足で実時間処理が難しくなる可能性がある。  
- ピン止めメモリ増で他プロセスと競合する可能性。  
- フィルタ係数の配布形式をどうするか（float32 を on-the-fly 拡張で済ませるか、double 版を用意するか）の運用判断が必要。

