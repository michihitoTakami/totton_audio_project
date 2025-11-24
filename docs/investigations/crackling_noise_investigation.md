# クラックリングノイズ調査報告 (Crackling Noise Investigation)

## 概要 (Summary)

GPU Audio Upsamplerにおいて、持続的なクラックリングノイズ("プチプチ音")が発生する問題を調査しました。
Overlap-Saveアルゴリズムのオーバーラップバッファ計算に不具合があり、修正により**77%のクリック削減**(48→11クリック)と**10dB改善**(-21.3dB→-30.7dB)を達成しましたが、完全には解消していません。

## 問題の特徴 (Problem Characteristics)

### ユーザー報告
- **症状**: 音声は途切れずに再生されるが、その上に規則的なクラックリングノイズが重畳
- **発生パターン**: ほぼ常時発生(音声再生中常に)
- **Integer変換の疑い**: オフラインWAV変換では正常動作するため、実時間処理時の問題と推測

### 技術的詳細
- **入力**: 44.1kHz stereo (PipeWire経由)
- **出力**: 352.8kHz stereo (8x upsampling, ALSA direct to SMSL D400EX DAC)
- **フィルタ**: 2,000,000タップ minimum-phase FIR
- **FFTサイズ**: 1,048,576サンプル
- **オーバーラップサイズ**: 999,999サンプル
- **有効出力/ブロック**: 48,577サンプル

## 波形解析結果 (Waveform Analysis)

### 解析手法
`scripts/analyze_waveform.py`を作成し、`test_data/fanfare.wav`(44.1kHz, 16-bit, 11.68秒)をGPU変換して検証:

```bash
./build/gpu_upsampler test_data/fanfare.wav test_data/fanfare_352800hz.wav --ratio 8
uv run python scripts/analyze_waveform.py test_data/fanfare_352800hz.wav
```

### 修正前の結果 (Before Fix)
```
検出されたクリック数 (Threshold: -40dB): 48
  最大振幅: -21.3dB (サンプル位置 23616516)
  平均振幅: -42.6dB

検出されたクリック数 (Threshold: -60dB): 95

クリック間隔の統計:
  平均: 8500754 samples (24.09秒)
  中央値: 8582144 samples
  最頻値: 48576-48577 samples ← **FFT有効出力サイズと一致!**
```

**重要な発見**: クリック間隔48576~48577サンプルは、Overlap-Saveの`validOutputPerBlock`サイズとぴったり一致。これはブロック境界での不連続性を示唆。

## 根本原因 (Root Cause)

### 問題箇所
`src/convolution_engine.cu` 行477-493 (修正前):

```cpp
// Save overlap for next block (last overlapSize_ samples of current block input)
if (outputPos + validOutputSize < outputFrames) {
    size_t overlapSourcePos = inputPos + validOutputSize;  // ← 誤り
    if (overlapSourcePos + overlapSize_ <= outputFrames) {
        Utils::checkCudaError(
            cudaMemcpyAsync(overlapBuffer.data(), d_upsampledInput + overlapSourcePos,
                           overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpy overlap from device"
        );
    }
}
```

### アルゴリズムの誤り
Overlap-Saveアルゴリズムでは:
- 各イテレーションの入力: `[前回オーバーラップ (999,999) | 新規サンプル (48,577)]`
- 出力: 最初の999,999サンプルを破棄、最後の48,577を保持

次回のオーバーラップは**現在ブロックの最後の999,999サンプル**であるべきだが、誤った計算により不連続が発生。

### 修正内容
`src/convolution_engine.cu` 行477-520 (修正後):

```cpp
// Save overlap for next block
if (outputPos + validOutputSize < outputFrames) {
    size_t nextBlockStart = inputPos + validOutputSize;
    if (nextBlockStart >= overlapSize_ && nextBlockStart < outputFrames) {
        size_t overlapStart = nextBlockStart - overlapSize_;  // ← 修正
        if (overlapStart + overlapSize_ <= outputFrames) {
            Utils::checkCudaError(
                cudaMemcpyAsync(overlapBuffer.data(), d_upsampledInput + overlapStart,
                               overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpy overlap from device"
            );
        }
    }
}
```

**キーポイント**: オーバーラップ開始位置を`nextBlockStart - overlapSize_`に修正し、正しい連続性を確保。

## 修正結果 (Fix Results)

### 波形解析比較

#### 修正後 (After Fix)
```bash
./build/gpu_upsampler test_data/fanfare.wav test_data/fanfare_352800hz_fixed.wav --ratio 8
uv run python scripts/analyze_waveform.py test_data/fanfare_352800hz_fixed.wav
```

```
検出されたクリック数 (Threshold: -40dB): 11 ← 77%減少(48→11)
  最大振幅: -30.7dB ← 10dB改善(-21.3dB→-30.7dB)
  平均振幅: -49.8dB

検出されたクリック数 (Threshold: -30dB): 0 ← 完全消失
検出されたクリック数 (Threshold: -60dB): 37 ← 61%減少(95→37)
```

### 実時間再生テスト
修正版デーモンでの再生結果:
- ✅ システム正常動作: PipeWire → GPU Upsampler → ALSA → SMSL D400EX (352.8kHz)
- ⚠️ クリッピング警告: 極少数のサンプル(~0.0003%)でクリッピング検出
- **ユーザー評価**: "あんまりプチプチは変わっていない"

## 残存問題 (Remaining Issues)

### 1. クラックリングノイズ依然として存在
**症状**: Overlap-Save修正後も、ユーザーには依然として明確なクラックリングノイズが聴こえる。

**可能性のある原因**:

#### A. Overlap-Saveアルゴリズムの更なる問題
- **位相連続性**: 現在の修正は振幅連続性のみ保証。位相の不連続が残っている可能性
- **境界条件**: 最初/最後のブロックでの特殊処理が不十分
- **検証**: `convolution_engine.cu`のConvolveBlock()メソッド全体のレビューが必要

#### B. バッファ管理の問題
**PipeWire → GPU間**:
```cpp
// src/alsa_daemon.cpp (行数は要確認)
void on_process(void* userdata) {
    // PipeWire callback: リングバッファに書き込み
}

void alsaOutputThread() {
    // ALSA出力スレッド: リングバッファから読み出し
}
```
- **リングバッファ同期**: Mutex保護は存在するが、読み書きポインタの不整合の可能性
- **アンダーラン/オーバーラン**: バッファ枯渇/溢れ時の処理が不適切

**GPU → ALSA間**:
- **メモリ転送タイミング**: cudaMemcpyAsyncの完了待機が不適切
- **ストリーム同期**: CUDA streamの同期ポイントが不足

#### C. サンプルレート変換の問題
- **8x upsampling精度**: 単純なゼロ挿入 + FIR畳み込みでは高周波ノイズが混入する可能性
- **タイムスタンプずれ**: PipeWireのクォンタムサイズとGPU処理ブロックサイズの不一致

#### D. 数値精度の問題
- **Float → Int32変換**: 既に疑われ調査済みだが、再検証の価値あり
  - `src/alsa_daemon.cpp`でのfloatToS32()変換処理
  - 丸め誤差の累積
  - クリッピング処理(-1.0~1.0の範囲外値の扱い)

#### E. ハードウェア/ドライバ問題
- **ALSA xrun**: カーネルログ(`dmesg | grep -i xrun`)で確認
- **USB DAC buffer**: SMSL D400EXのUSBバッファサイズ(現在131,072 frames)が大きすぎる/小さすぎる
- **DMAタイミング**: PCIe DMA転送とUSB転送のタイミング競合

### 2. クリッピング警告
```
WARNING: Clipping detected - 95 samples clipped out of 29163520 (0.000325749%)
```

**影響**: 0.0003%程度のサンプルで±1.0超え。人間の可聴域では検出困難だが、ピーク時の歪みの可能性。

**原因候補**:
- フィルタ係数の正規化不足
- アップサンプリング時のゲイン補正ミス
- 数値誤差の累積

## 推奨される次のステップ (Recommended Next Steps)

### 優先度1: アルゴリズム検証 (最重要)
1. **Overlap-Save完全検証**
   ```bash
   # デバッグ版ビルド
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make gpu_upsampler

   # 詳細ログ出力でブロック境界の連続性を確認
   # convolution_engine.cu に以下を追加:
   printf("Block %zu: inputPos=%zu, outputPos=%zu, overlap[0]=%f, overlap[last]=%f\n",
          blockIdx, inputPos, outputPos, overlapBuffer[0], overlapBuffer[overlapSize_-1]);
   ```

2. **位相連続性の検証**
   - 連続するブロックのFFT係数の位相差を確認
   - Overlap-Add方式との比較実装

3. **境界条件テスト**
   - 非常に短い入力(1ブロック未満)での動作確認
   - ブロック境界またがりのインパルス応答テスト

### 優先度2: バッファ管理の詳細調査
1. **リングバッファのトレース**
   ```cpp
   // src/alsa_daemon.cpp に追加
   std::ofstream trace("buffer_trace.log", std::ios::app);
   trace << timestamp << " Write:" << writePos << " Read:" << readPos
         << " Avail:" << available << std::endl;
   ```

2. **アンダーラン/オーバーラン検出**
   - PipeWireコールバックとALSAスレッドの実行頻度を測定
   - バッファ利用率のヒストグラム作成

### 優先度3: サンプルレート変換の再検証
1. **クリッピング原因の特定**
   ```bash
   # フィルタ係数の分析
   uv run python -c "
   import numpy as np
   coeffs = np.fromfile('data/coefficients/filter_44k_2m_min_phase.bin', dtype=np.float32)
   print(f'Sum: {coeffs.sum()}, Max: {coeffs.max()}, Min: {coeffs.min()}')
   print(f'Energy: {(coeffs**2).sum()}')
   "
   ```
   - フィルタゲインが1.0以外なら正規化

2. **代替アップサンプリング手法のプロトタイプ**
   - Polyphase decomposition
   - Farrow構造

### 優先度4: Float→Int32変換の再検証
```cpp
// src/alsa_daemon.cpp の floatToS32() を詳細ログ付きで実装
int32_t floatToS32(float sample) {
    if (std::isnan(sample) || std::isinf(sample)) {
        fprintf(stderr, "Invalid sample: %f\n", sample);
        return 0;
    }
    // Soft clipping (tanh) を試す
    sample = std::tanh(sample);

    // 丸め方向を明示
    return static_cast<int32_t>(std::round(sample * 2147483647.0f));
}
```

### 優先度5: ハードウェア/ドライバ確認
```bash
# ALSA xrun確認
dmesg | grep -i "xrun\|underrun\|overrun"

# USBデバイス情報
lsusb -v -d <SMSL_VENDOR_ID>:<SMSL_PRODUCT_ID> | grep -i "interval\|buffer"

# リアルタイム優先度設定(既存だが再確認)
chrt -f -p 99 $(pgrep gpu_upsampler_alsa)
```

## 変更されたファイル (Modified Files)

### コア修正
- `/home/michihito/Working/gpu_os/src/convolution_engine.cu` (行477-520): Overlap-Saveアルゴリズム修正

### 解析ツール
- `/home/michihito/Working/gpu_os/scripts/analyze_waveform.py` (新規作成): クリック検出・波形解析
- `/home/michihito/Working/gpu_os/pyproject.toml` (行11): soundfile依存追加

### テストデータ
- `test_data/fanfare_352800hz.wav`: 修正前の出力
- `test_data/fanfare_352800hz_fixed.wav`: 修正後の出力
- `test_data/fanfare_352800hz_analysis.png`: 修正前の解析結果
- `test_data/fanfare_352800hz_fixed_analysis.png`: 修正後の解析結果

## システム構成 (System Configuration)

### ハードウェア
- **DAC**: SMSL D400EX (USB, 最大768kHz対応)
- **GPU**: NVIDIA GeForce RTX 2070 Super (8GB VRAM)
- **ALSA カード**: card 3, device 0

### ソフトウェア
- **OS**: Linux (カーネル要確認)
- **PipeWire**: バージョン要確認
- **CUDA**: バージョン要確認

### 音声経路 (Audio Path)
```
[アプリケーション(Spotify等)]
        ↓ (PulseAudio/PipeWire)
[gpu_upsampler_sink] (PipeWire null sink)
        ↓ monitor
[GPU Upsampler Input] (PipeWire stream)
        ↓ (リングバッファ)
[GPU Processing] (CUDA, 44.1kHz → 352.8kHz, 1M-tap FIR)
        ↓
[ALSA Output] (S32_LE, 352.8kHz, 2ch, buffer 131072 frames)
        ↓ (USB)
[SMSL D400EX DAC]
        ↓ (アナログ)
[スピーカー/ヘッドフォン]
```

### PipeWire接続 (確認済み)
```bash
pw-link -l
# 正常接続:
# spotify:output_FL → gpu_upsampler_sink:playback_FL
# spotify:output_FR → gpu_upsampler_sink:playback_FR
# gpu_upsampler_sink:monitor_FL → GPU Upsampler Input:input_FL
# gpu_upsampler_sink:monitor_FR → GPU Upsampler Input:input_FR
```

## 参考情報 (References)

### Overlap-Save理論
- [Wikipedia: Overlap–save method](https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method)
- [DSPGuide Chapter 18: FFT Convolution](http://www.dspguide.com/ch18.htm)

### 関連Issue/PR
- (適宜追加)

## 作成者・日時 (Author & Date)
- **作成日**: 2025-11-21
- **最終調査者**: Claude Code (GPT-4 based AI assistant)
- **ユーザーフィードバック**: michihito

---

**結論**: Overlap-Saveの基本的な不具合は修正され、測定可能な改善(77%クリック削減)を達成。しかし、ユーザー体感では依然としてクラックリングノイズが残存。次の調査者は**優先度1のアルゴリズム完全検証**から着手することを強く推奨します。
