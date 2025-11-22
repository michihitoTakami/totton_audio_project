# Phase 2 実装報告書：GPU FFT畳み込みエンジン（プロトタイプ版）

## 実装期間
2025年11月21日

## 目的
131,072タップ最小位相FIRフィルタを用いた16倍アップサンプリング（44.1kHz → 705.6kHz）のスタンドアロンC++アプリケーションの実装。

---

## 1. 実装概要

### 1.1 技術スタック

| 項目 | 採用技術 | 選定理由 |
|------|----------|----------|
| GPU API | CUDA 12.0 + cuFFT | 開発速度優先、RTX 2070S最適化 |
| WAV I/O | libsndfile 1.2.2 | 業界標準、Hi-Res対応 |
| ビルドシステム | CMake 3.28 | CUDA統合が容易 |
| FFT方式 | Overlap-Save法（設計） | ステレオ処理との相性良好 |

### 1.2 成果物

```
gpu_os/
├── CMakeLists.txt                    # ビルド設定
├── include/
│   ├── audio_io.h                    # WAV I/Oインターフェース
│   └── convolution_engine.h          # GPU畳み込みエンジンAPI
├── src/
│   ├── main.cpp                      # CLIアプリケーション
│   ├── audio_io.cpp                  # libsndfileラッパー実装
│   └── convolution_engine.cu         # CUDAカーネル実装
├── build/
│   └── gpu_upsampler                 # 実行ファイル（80KB）
├── scripts/
│   └── generate_test_audio.py        # テスト信号生成ツール
└── test_data/
    ├── test_sine_1khz_44100hz.wav    # 入力テスト信号
    └── output_sine_1khz_705600hz.wav # 出力（705.6kHz）
```

---

## 2. 機能実装詳細

### 2.1 WAV I/Oモジュール (`audio_io.cpp/.h`)

#### 実装機能
- **WavReader**: libsndfileラッパー（全データ読み込み / ブロック読み込み）
- **WavWriter**: 32bit float WAV出力（Hi-Resサンプルレート対応）
- **Utils関数**:
  - `interleavedToSeparate`: ステレオ → L/R分離
  - `separateToInterleaved`: L/R → ステレオ合成
  - `monoToStereo`, `stereoToMono`: チャンネル変換

#### テスト結果
- ✅ 44.1kHz mono/stereo WAV読み込み成功
- ✅ 705.6kHz stereo WAV出力成功（5.4MB）

### 2.2 GPU畳み込みエンジン (`convolution_engine.cu/.h`)

#### 実装コンポーネント

##### CUDAカーネル
1. **zeroPadKernel**: ゼロ詰めによるアップサンプリング
   ```cuda
   output[idx * upsampleRatio] = input[idx];
   // 中間サンプルは0で埋める
   ```

2. **complexMultiplyKernel**: 周波数領域での複素数乗算
   ```cuda
   data[idx] = data[idx] * filter[idx]  // 複素数演算
   ```

3. **scaleKernel**: IFFT後のスケーリング補正

##### GPU リソース管理
- **フィルタ係数**: 512KB（131kタップ）をGPU VRAMに常駐
- **FFTプラン**: cuFFT R2C/C2R（サイズ262,144 = 2の次のべき乗）
- **ワーキングバッファ**: 入力/出力/FFT結果用

#### 現在の実装状態：プロトタイプ版

**重要な注意事項:**
現在のバージョンは**プロトタイプ実装**です。`convolution_engine.cu:182`に記載の通り、FFT畳み込みはCPU側で単純なtime-domain畳み込みとして実装されています。

```cpp
// Simple time-domain convolution on CPU (for prototype)
// TODO: Replace with full GPU FFT convolution for production
for (size_t i = 0; i < outputFrames; ++i) {
    double sum = 0.0;
    for (int j = 0; j < filterTaps_ && i >= static_cast<size_t>(j); ++j) {
        sum += h_upsampled[i - j] * h_filterCoeffs_[j];
    }
    outputData[i] = static_cast<float>(sum);
}
```

**計算量:** O(N × M)（N=出力フレーム数、M=131kタップ）
→ 1秒の音声で **705,600 × 131,072 ≈ 925億回の乗算**

### 2.3 メインアプリケーション (`main.cpp`)

#### CLI仕様
```bash
./build/gpu_upsampler <input.wav> <output.wav> [options]

Options:
  --filter <path>   フィルタ係数ファイル（default: data/coefficients/filter_131k_min_phase.bin）
  --ratio <n>       アップサンプリング倍率（default: 16）
  --block <size>    処理ブロックサイズ（default: 8192）
```

#### 処理パイプライン
1. **入力読み込み**: libsndfileで全データ読み込み
2. **GPU初期化**: フィルタ係数ロード、cuFFT準備
3. **チャンネル処理**:
   - Mono: 1チャンネル処理 → ステレオ複製
   - Stereo: L/R分離 → 個別処理 → 合成
4. **出力書き込み**: 705.6kHz stereo WAV

---

## 3. 動作確認テスト

### 3.1 テスト環境
- **GPU**: NVIDIA GeForce RTX 2070 Super（8GB VRAM）
- **Driver**: 570.195.03
- **CUDA**: 12.0.140
- **OS**: Ubuntu 24.04

### 3.2 テスト信号

| 信号種類 | 周波数 | サンプルレート | 長さ | ファイルサイズ |
|---------|--------|--------------|------|--------------|
| 1kHz正弦波 | 1000 Hz | 44.1 kHz | 1秒 | 176 KB |
| 10kHz正弦波 | 10000 Hz | 44.1 kHz | 1秒 | 176 KB |
| 周波数スイープ | 20-20k Hz | 44.1 kHz | 1秒 | 176 KB |
| インパルス | - | 44.1 kHz | 1秒 | 176 KB |
| ステレオテスト | 1k/2k Hz | 44.1 kHz | 1秒 | 352 KB |

### 3.3 実行結果（1kHz正弦波）

```
Input:  44100 frames @ 44100 Hz (1.00 sec)
Output: 705600 frames @ 705600 Hz

Performance:
  Processing time: 81.11 sec
  Total time:      81.57 sec
  Speed:           0.01x realtime

Output file: test_data/output_sine_1khz_705600hz.wav (5.4MB)
```

#### 検証項目
- ✅ **正常動作**: 出力ファイル生成成功
- ✅ **サンプルレート**: 705,600 Hz（16倍）
- ✅ **ファイルサイズ**: 5.4MB（705600 × 2ch × 4byte = 理論値一致）
- ⚠️ **処理速度**: 0.01倍リアルタイム（目標: 10倍以上）

---

## 4. 性能分析

### 4.1 現在の性能（プロトタイプ版）

| 指標 | 実測値 | 目標値 | 達成率 |
|------|--------|--------|--------|
| 処理速度 | 0.01x RT | > 10x RT | **0.1%** |
| GPU利用率 | ~0%（CPU処理） | < 20% | - |
| レイテンシ | 81秒/秒音声 | < 10ms | - |

### 4.2 ボトルネック分析

#### CPU time-domain畳み込みの計算量
- **乗算回数**: 705,600フレーム × 131,072タップ = **92,559,667,200回**（約925億回）
- **処理時間**: 81.11秒
- **スループット**: 約1.14億回/秒の乗算

#### GPU FFT畳み込みの理論性能（未実装）
- **FFT計算量**: O(N log N) ≈ 705,600 × log₂(262,144) ≈ 12.7M ops
- **複素数乗算**: 131,072回（周波数領域）
- **理論処理時間**: < 10ms（RTX 2070S = 7.5 TFLOPs）

**速度改善比**: GPU実装により **約8,000倍以上の高速化** が期待される

---

## 5. Phase 2の達成状況

### 5.1 完了項目 ✅

1. **環境構築**
   - CUDA Toolkit 12.0インストール
   - libsndfile開発パッケージセットアップ
   - CMakeビルドシステム構築

2. **コアモジュール実装**
   - WAV I/Oラッパー（読み書き、チャンネル変換）
   - CUDAカーネル基盤（ゼロ詰め、複素数演算、スケーリング）
   - GPUリソース管理（メモリ確保、cuFFTプラン）

3. **CLIアプリケーション**
   - コマンドライン引数パース
   - 進捗表示
   - 統計情報出力

4. **動作確認**
   - ビルド成功（警告なし）
   - テスト信号生成
   - 実ファイル変換成功

### 5.2 未完了項目（今後の最適化が必要）⚠️

1. **GPU FFT畳み込みの完全実装**
   - 現状: CPU time-domain（0.01x RT）
   - 目標: GPU FFT Overlap-Save（> 10x RT）
   - 作業: `convolution_engine.cu:processChannel`を完全GPU化

2. **Partitioned FFT実装**
   - 長時間音声対応（ブロック分割処理）
   - Overlap-Save/Addアルゴリズム
   - リングバッファ管理

3. **非同期処理**
   - CUDA Streamによるパイプライン化
   - H2D/D2H転送とカーネル実行のオーバーラップ

4. **性能計測**
   - GPU負荷率モニタリング（NVML統合）
   - 詳細プロファイリング（nvprof/Nsight Systems）

5. **検証**
   - 周波数特性解析（FFT出力との比較）
   - SNR/THD測定
   - 音質主観評価

---

## 6. Phase 3への引き継ぎ事項

### 6.1 アーキテクチャ設計

現在の実装は **「機能は動作するが、性能が出ていない」** 状態です。Phase 3（LV2プラグイン化）に進む前に、以下の最適化が必須です。

#### 優先度1: GPU FFT畳み込みの完全実装
**箇所**: `src/convolution_engine.cu:processChannel()`

**現在の疑似コード:**
```
1. ゼロ詰め（GPU）
2. GPU → CPU転送
3. time-domain畳み込み（CPU）← ボトルネック
4. CPU → GPU転送
```

**目標アーキテクチャ:**
```
1. ゼロ詰め（GPU）
2. FFT（GPU）
3. 複素数乗算（GPU）
4. IFFT（GPU）
5. Overlap-Save処理（GPU）
```

#### 優先度2: Partitioned FFT
長時間音声（数分〜数十分）に対応するため、ブロック分割処理が必須。

**実装方針:**
- ブロックサイズ: 8192サンプル（約185ms @44.1kHz）
- オーバーラップ: 131,071サンプル（フィルタ長-1）
- メモリ効率: GPU上でリングバッファ管理

#### 優先度3: ステレオ並列処理
現在はL/R逐次処理。CUDA Streamで並列化可能。

```cpp
cudaStream_t streamL, streamR;
// L/R同時処理
processChannelAsync(left, streamL);
processChannelAsync(right, streamR);
cudaDeviceSynchronize();
```

### 6.2 パフォーマンス目標

| 指標 | Phase 2プロトタイプ | Phase 2最適化版（目標） | Phase 3リアルタイム（目標） |
|------|---------------------|----------------------|---------------------------|
| 処理速度 | 0.01x RT | **> 50x RT** | > 1x RT |
| GPU利用率 | ~0% | **10-15%** | < 20% |
| レイテンシ | 81秒 | < 100ms | 50-100ms |

### 6.3 Phase 3で対応すべき項目

1. **LV2プラグイン化**
   - LV2 API実装（Atom、Worker拡張）
   - サンプルレート変更対応
   - レイテンシ報告機能

2. **PipeWire統合**
   - リングバッファ（音切れ防止）
   - サンプルレートネゴシエーション
   - Easy Effectsでの動作確認

3. **エラーハンドリング**
   - GPU Out of Memory対応
   - サンプルレート不一致の処理
   - プラグイン動的ロード/アンロード

---

## 7. 技術的考察

### 7.1 なぜGPU FFT畳み込みが必要か

#### Time-domain畳み込みの限界
- **計算量**: O(N × M) = N個のサンプルとM個のタップの総当たり
- **131kタップの場合**: 1秒音声で約925億回の乗算
- **マルチコア最適化しても**: 現代のCPUでは実時間処理不可能

#### FFT畳み込みの利点
- **計算量削減**: O(N log N) ≈ 1270万回（約7,300倍削減）
- **GPUの強み**: 数千コアで並列FFT実行
- **cuFFT最適化**: NVIDIAがチューニング済み

### 7.2 Overlap-Save vs Overlap-Add

#### 本プロジェクトでの選択: **Overlap-Save推奨**

| 観点 | Overlap-Save | Overlap-Add |
|------|--------------|-------------|
| メモリ効率 | ○ 入力側オーバーラップ | △ 出力側加算バッファ必要 |
| 実装複雑度 | ○ シンプル | △ 加算処理追加 |
| ステレオ対応 | ○ 独立処理可能 | △ 出力バッファ管理複雑 |

### 7.3 RTX 2070 Superの活用

#### ハードウェアスペック
- **CUDAコア**: 2,560個
- **Compute Capability**: 7.5（Turing世代）
- **VRAM**: 8GB GDDR6
- **Tensor Core**: 320個（今回は未使用）

#### メモリ配分（推定）
- フィルタ係数: 512 KB
- FFTワーキングバッファ: 約10 MB（ブロック処理時）
- 余裕: 7.98 GB（長時間音声にも対応可能）

---

## 8. 今後の開発ロードマップ

### Phase 2 最適化版（推奨される次ステップ）

#### ステップ1: GPU FFT畳み込み実装（2-3日）
1. `processChannel()`内のCPU畳み込みをGPU FFTに置き換え
2. cuFFT R2C → 複素数乗算 → C2R パイプライン構築
3. 単一ブロック処理での動作確認

**期待される成果:**
- 処理速度: 0.01x → 50x RT（5,000倍高速化）
- GPU利用率: 0% → 10-15%

#### ステップ2: Partitioned FFT実装（2-3日）
1. ブロック分割ロジック追加
2. Overlap-Save境界処理
3. 長時間音声テスト（10分以上）

**期待される成果:**
- メモリ使用量の最適化
- 任意長音声の処理可能

#### ステップ3: 非同期処理（1-2日）
1. CUDA Streamによるパイプライン化
2. ステレオL/R並列処理
3. H2D/D2H転送の最適化

**期待される成果:**
- 処理速度: 50x → 100x RT
- GPU利用率: 15% → 20%

#### ステップ4: 検証（1日）
1. 周波数特性測定（Python解析）
2. GPU負荷率モニタリング
3. 音質主観評価

### Phase 3: LV2プラグイン化（Phase 2最適化後）

詳細は`docs/first.txt`のPhase 3仕様参照。

---

## 9. 参考資料

### 9.1 実装に使用したリソース

- **cuFFT Documentation**: https://docs.nvidia.com/cuda/cufft/
- **libsndfile API**: http://www.mega-nerd.com/libsndfile/api.html
- **Overlap-Save Algorithm**: Oppenheim & Schafer, "Discrete-Time Signal Processing"

### 9.2 性能最適化の参考

- **NVIDIA CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **FFT Convolution on GPU**: "GPU Gems 3" Chapter 18

---

## 10. 付録：ビルドと実行手順

### 10.1 初回ビルド

```bash
# 依存パッケージインストール（要sudo）
sudo apt update
sudo apt install -y cmake build-essential libsndfile1-dev nvidia-cuda-toolkit

# ビルド
cd /home/michihito/Working/gpu_os
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 10.2 テスト信号生成

```bash
cd /home/michihito/Working/gpu_os
uv run python scripts/generate_test_audio.py --duration 1.0
```

### 10.3 アップサンプリング実行

```bash
./build/gpu_upsampler \
    test_data/test_sine_1khz_44100hz.wav \
    test_data/output_sine_1khz_705600hz.wav
```

### 10.4 出力確認（Audacityなど）

```bash
# サンプルレート確認
soxi test_data/output_sine_1khz_705600hz.wav
```

---

## 11. まとめ

Phase 2プロトタイプ版では、以下を達成しました：

✅ **CUDA + cuFFT環境構築**
✅ **WAV I/Oモジュール実装**（libsndfile）
✅ **CUDAカーネル基盤実装**（ゼロ詰め、複素数演算）
✅ **CLIアプリケーション完成**
✅ **実ファイル変換成功**（44.1kHz → 705.6kHz）

⚠️ **次の最適化が必要:**
- GPU FFT畳み込みの完全実装（現在CPU処理のため遅い）
- Partitioned FFT実装
- 性能検証

**Phase 2最適化版の実装により、処理速度は5,000倍以上に向上し、Phase 3（リアルタイムLV2プラグイン）への準備が整います。**
