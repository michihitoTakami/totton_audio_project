# Overlap-Save アルゴリズム修正計画 (Fix Plan)

## 作成日時
2025-11-21

## 問題の発見経緯

前回のセッションで、サンプルレートを8xから16xに変更し、ソフトクリッピング(tanh)を導入し、デバッグログを追加しました。その結果、以下の致命的な問題が発見されました:

```
[DEBUG] Block 0: inputPos=0, outputPos=0, validOutputSize=16384, currentBlockSize=16384
[DEBUG] Block 0: inputPos=0, outputPos=0, validOutputSize=16384, currentBlockSize=16384
[DEBUG] Block 0: inputPos=0, outputPos=0, validOutputSize=16384, currentBlockSize=16384
...
```

**無限ループ状態**: Block 0が永遠に繰り返され、inputPos/outputPosが全く進んでいません。

## 根本原因の特定

### 症状
- `blockCount`: 常に0 (インクリメントされない)
- `inputPos`: 常に0 (進まない)
- `outputPos`: 常に0 (進まない)
- `validOutputSize`: 16384 (期待値: 48,577)
- `currentBlockSize`: 16384 (期待値: 48,577)

### 原因箇所

#### 問題1: `validOutputSize`の誤った計算 (行477-478)

```cpp
// src/convolution_engine.cu:477-478
size_t validOutputSize = (outputFrames - outputPos < static_cast<size_t>(validOutputPerBlock)) ?
                          (outputFrames - outputPos) : static_cast<size_t>(validOutputPerBlock);
```

**誤りの分析**:
- この計算は「出力バッファの残りフレーム数」を基準にしています
- `outputFrames`は出力バッファの総サイズ(= `inputFrames * upsampleRatio_`)
- `validOutputPerBlock`は48,577と正しく計算されています(行390)
- しかし、`outputPos`が0の状態で、もし`outputFrames`が小さい場合、または何らかの理由で三項演算子が常に最初の分岐を選んでいる場合、16,384という値が選ばれます

**16,384の正体**:
- これはALSAの**Period size**です(`ALSA: Period size: 16384 frames`)
- どこかで`outputFrames`が誤って16,384にセットされている可能性があります
- または、`ConvolveBlock()`が**リアルタイム処理コンテキスト**で呼ばれており、`outputFrames`が1ブロック分(16,384)しか確保されていない可能性があります

#### 問題2: ブロック進行の失敗 (行543-545)

```cpp
// src/convolution_engine.cu:543-545
outputPos += validOutputSize;  // 0 + 16384 = 16384
inputPos += validOutputSize;   // 0 + 16384 = 16384
blockCount++;                  // 0 + 1 = 1
```

この計算自体は正しいですが、`validOutputSize`が16,384の場合:
- `outputPos`は16,384に進みます
- 次のイテレーションでループ条件`while (outputPos < outputFrames)`を評価
- **もし`outputFrames == 16384`なら、ループを抜けてしまいます**
- しかし、ログでは「Block 0」が無限に繰り返されています

**これが意味すること**:
- `ConvolveBlock()`が**毎回新しく呼ばれている**可能性があります
- リアルタイム処理では、PipeWireから4,096サンプルが来るたびに`ConvolveBlock()`が呼ばれます
- そのたびに、`inputPos`と`outputPos`が0にリセットされています
- つまり、**`ConvolveBlock()`はストリーミング処理用に設計されていない**のです

### 設計上の根本的な誤り

`ConvolveBlock()`メソッドは、**オフラインWAV変換用**に設計されています:
```cpp
void ConvolveBlock(const float* inputData, size_t inputFrames,
                   std::vector<float>& outputData, cudaStream_t stream)
```

この関数は:
1. 入力全体(`inputData`、`inputFrames`サンプル)を一度に処理する前提です
2. ゼロパディング→アップサンプリング→Overlap-Save FFT convolutionを順に実行します
3. 出力バッファ(`outputData`)に全結果を書き込みます

しかし、**リアルタイムストリーミング処理**では:
1. PipeWireから**4,096サンプルずつ**データが到着します
2. 毎回新しい`ConvolveBlock()`呼び出しが発生します
3. 各呼び出しで`inputPos`/`outputPos`が0にリセットされます
4. Overlap-Saveのオーバーラップバッファは保持されますが、ブロック進行ロジックが機能しません

## 修正方針

### Option A: リアルタイム用Overlap-Save実装の追加 (推奨)

`ConvolveBlock()`をオフライン処理専用として残し、**新しいストリーミング用メソッド**を追加します:

```cpp
// 新規メソッド
void ConvolveStreamBlock(const float* inputData, size_t inputFrames,
                         std::vector<float>& outputData, cudaStream_t stream);
```

**設計のポイント**:
1. **状態の保持**: クラスメンバ変数として、ストリーミング処理のための状態を保持
   ```cpp
   // Streaming state (add to ConvolutionEngine class)
   std::vector<float> streamInputBuffer_;     // 蓄積した入力サンプル
   size_t streamInputAccumulated_;            // 蓄積済みサンプル数
   size_t streamOutputGenerated_;             // 生成済み出力サンプル数
   ```

2. **バッファリング**: 入力が`validOutputPerBlock / upsampleRatio_`(= 48,577 / 16 = 3,036サンプル)に達するまで蓄積
   - PipeWireから4,096サンプルが来る
   → バッファに蓄積
   → 3,036サンプル以上溜まったら処理開始

3. **Overlap-Save処理**:
   - 蓄積バッファから3,036サンプル取り出し
   - ゼロパディング → アップサンプリング(48,577サンプル)
   - `[overlapBuffer (999,999) | 新規サンプル (48,577)]`でFFT convolution
   - 最初の999,999サンプル破棄、最後の48,577を出力
   - 次回のオーバーラップを保存

4. **出力**: 48,577サンプル(705.6kHz)を返す
   - これをリングバッファに書き込み
   - ALSA出力スレッドが読み出す

### Option B: `ConvolveBlock()`の修正 (非推奨)

現在の`ConvolveBlock()`を修正してストリーミング対応させる方法もありますが:
- オフライン処理(WAV変換)が壊れるリスクあり
- コードが複雑化する
- 後方互換性の維持が困難

そのため、Option Aの「新規メソッド追加」を推奨します。

## 実装計画

### Phase 1: ストリーミングバッファの追加

**ファイル**: `src/convolution_engine.h`

```cpp
class ConvolutionEngine {
private:
    // ... existing members ...

    // Streaming state
    std::vector<float> streamInputBuffer_;        // Accumulated input samples
    size_t streamInputAccumulated_;               // Number of samples in buffer
    size_t streamValidInputPerBlock_;             // Input samples needed per block
    bool streamInitialized_;                      // Whether streaming mode is active

public:
    // Initialize streaming mode
    void InitializeStreaming();

    // Reset streaming state
    void ResetStreaming();

    // Process one streaming block
    void ConvolveStreamBlock(const float* inputData, size_t inputFrames,
                             std::vector<float>& outputData, cudaStream_t stream);
};
```

### Phase 2: ストリーミングメソッドの実装

**ファイル**: `src/convolution_engine.cu`

```cpp
void ConvolutionEngine::InitializeStreaming() {
    if (!initialized_) {
        throw std::runtime_error("GPU resources not initialized");
    }

    streamValidInputPerBlock_ = validOutputPerBlock_ / upsampleRatio_;
    // For 16x upsampling: 48577 / 16 = 3036 samples (at 44.1kHz)

    streamInputBuffer_.resize(streamValidInputPerBlock_ * 2);  // 2x buffer for safety
    streamInputAccumulated_ = 0;
    streamInitialized_ = true;

    fprintf(stderr, "[Streaming] Initialized: need %zu input samples per block\n",
            streamValidInputPerBlock_);
}

void ConvolutionEngine::ResetStreaming() {
    streamInputAccumulated_ = 0;
    std::fill(streamInputBuffer_.begin(), streamInputBuffer_.end(), 0.0f);
    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);
}

void ConvolutionEngine::ConvolveStreamBlock(const float* inputData, size_t inputFrames,
                                             std::vector<float>& outputData, cudaStream_t stream) {
    if (!streamInitialized_) {
        throw std::runtime_error("Streaming mode not initialized");
    }

    // 1. Accumulate input samples
    if (streamInputAccumulated_ + inputFrames > streamInputBuffer_.size()) {
        throw std::runtime_error("Stream input buffer overflow");
    }

    std::copy(inputData, inputData + inputFrames,
              streamInputBuffer_.begin() + streamInputAccumulated_);
    streamInputAccumulated_ += inputFrames;

    // 2. Check if we have enough samples for one block
    if (streamInputAccumulated_ < streamValidInputPerBlock_) {
        // Not enough data yet - return empty output
        outputData.clear();
        return;
    }

    // 3. Process one block
    size_t samplesToProcess = streamValidInputPerBlock_;

    // Step 3a: Zero-padding + upsampling (same as ConvolveBlock)
    float* d_input = nullptr;
    float* d_upsampledInput = nullptr;
    size_t outputFrames = samplesToProcess * upsampleRatio_;

    Utils::checkCudaError(
        cudaMalloc(&d_input, samplesToProcess * sizeof(float)),
        "cudaMalloc streaming input"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_upsampledInput, outputFrames * sizeof(float)),
        "cudaMalloc streaming upsampled"
    );

    Utils::checkCudaError(
        cudaMemcpyAsync(d_input, streamInputBuffer_.data(), samplesToProcess * sizeof(float),
                       cudaMemcpyHostToDevice, stream),
        "cudaMemcpy streaming input to device"
    );

    int threadsPerBlock = 256;
    int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
    zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_input, d_upsampledInput, samplesToProcess, upsampleRatio_
    );

    cudaFree(d_input);

    // Step 3b: Overlap-Save FFT convolution (ONE BLOCK ONLY)
    float* d_paddedInput = nullptr;
    cufftComplex* d_inputFFT = nullptr;
    float* d_convResult = nullptr;

    Utils::checkCudaError(
        cudaMalloc(&d_paddedInput, fftSize_ * sizeof(float)),
        "cudaMalloc streaming padded input"
    );

    int fftComplexSize = fftSize_ / 2 + 1;
    Utils::checkCudaError(
        cudaMalloc(&d_inputFFT, fftComplexSize * sizeof(cufftComplex)),
        "cudaMalloc streaming input FFT"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_convResult, fftSize_ * sizeof(float)),
        "cudaMalloc streaming conv result"
    );

    // Prepare input: [overlap | new samples]
    Utils::checkCudaError(
        cudaMemsetAsync(d_paddedInput, 0, fftSize_ * sizeof(float), stream),
        "cudaMemset streaming padded"
    );

    if (overlapSize_ > 0) {
        Utils::checkCudaError(
            cudaMemcpyAsync(d_paddedInput, overlapBuffer_.data(),
                           overlapSize_ * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpy streaming overlap to device"
        );
    }

    Utils::checkCudaError(
        cudaMemcpyAsync(d_paddedInput + overlapSize_, d_upsampledInput,
                       outputFrames * sizeof(float), cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy streaming block to padded"
    );

    // FFT convolution
    Utils::checkCufftError(
        cufftExecR2C(fftPlanForward_, d_paddedInput, d_inputFFT),
        "cufftExecR2C streaming"
    );

    threadsPerBlock = 256;
    blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
    complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_inputFFT, d_filterFFT_, fftComplexSize
    );

    Utils::checkCufftError(
        cufftExecC2R(fftPlanInverse_, d_inputFFT, d_convResult),
        "cufftExecC2R streaming"
    );

    // Scale
    float scale = 1.0f / fftSize_;
    int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
        d_convResult, fftSize_, scale
    );

    // Extract valid output (discard first overlapSize_ samples)
    outputData.resize(validOutputPerBlock_);
    Utils::checkCudaError(
        cudaMemcpyAsync(outputData.data(), d_convResult + overlapSize_,
                       validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpy streaming output to host"
    );

    // Save overlap for next block
    if (outputFrames >= static_cast<size_t>(overlapSize_)) {
        size_t overlapStart = outputFrames - overlapSize_;
        Utils::checkCudaError(
            cudaMemcpyAsync(overlapBuffer_.data(), d_upsampledInput + overlapStart,
                           overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpy streaming overlap from device"
        );
    }

    // Synchronize to ensure completion
    Utils::checkCudaError(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize streaming"
    );

    // Cleanup
    cudaFree(d_paddedInput);
    cudaFree(d_inputFFT);
    cudaFree(d_convResult);
    cudaFree(d_upsampledInput);

    // 4. Shift remaining samples in input buffer
    size_t remaining = streamInputAccumulated_ - samplesToProcess;
    if (remaining > 0) {
        std::copy(streamInputBuffer_.begin() + samplesToProcess,
                  streamInputBuffer_.begin() + streamInputAccumulated_,
                  streamInputBuffer_.begin());
    }
    streamInputAccumulated_ = remaining;
}
```

### Phase 3: `alsa_daemon.cpp`の修正

**変更箇所**: GPU処理部分で`ConvolveStreamBlock()`を使用

```cpp
// src/alsa_daemon.cpp (GPU処理スレッドまたはコールバック内)

// 初期化時
upsampler.InitializeStreaming();

// PipeWireコールバック内
void on_process(void* userdata) {
    // ... existing code ...

    // リングバッファからGPU処理用バッファに読み込み
    std::vector<float> inputBlock(BLOCK_SIZE);
    // ... copy from ring buffer to inputBlock ...

    // GPU処理 (ステレオ各チャンネル)
    std::vector<float> outputLeft, outputRight;
    upsampler.ConvolveStreamBlock(inputBlock.data() + leftChannelOffset, BLOCK_SIZE,
                                   outputLeft, leftStream);
    upsampler.ConvolveStreamBlock(inputBlock.data() + rightChannelOffset, BLOCK_SIZE,
                                   outputRight, rightStream);

    // outputLeft/Right には validOutputPerBlock (48577) サンプルが入っている
    // または、まだ蓄積中の場合は空

    if (!outputLeft.empty() && !outputRight.empty()) {
        // インターリーブしてリングバッファに書き込み
        // ...
    }
}
```

### Phase 4: テストと検証

#### 4-1: オフライン変換の保証
既存のWAV変換が壊れていないことを確認:
```bash
./build/gpu_upsampler test_data/fanfare.wav test_data/fanfare_705600hz.wav --ratio 16
uv run python scripts/analyze_waveform.py test_data/fanfare_705600hz.wav
```

期待結果:
- クリック検出数: -40dBで11個以下(前回の修正後と同等)
- 波形に大きな不連続なし

#### 4-2: リアルタイム処理の検証
デーモンを起動してデバッグログを確認:
```bash
./build/gpu_upsampler_alsa 2>&1 | tee debug_streaming.log
```

期待されるログ:
```
[Streaming] Initialized: need 3036 input samples per block
[Streaming] Accumulated 4096 samples (need 3036) - processing 1 block
[Streaming] Block processed: 48577 output samples generated
[Streaming] Remaining input: 1060 samples
...
```

**成功の指標**:
- ブロックが正常に進む(Block 0, 1, 2, ... と増加)
- `validOutputSize = 48577`が表示される
- クラックリングノイズが大幅に減少または消失

#### 4-3: 波形解析
リアルタイム出力を録音して解析:
```bash
# 440Hz正弦波を再生
speaker-test -D gpu_upsampler_sink -c 2 -r 44100 -t sine -f 440 &

# ALSA出力を録音 (別ターミナル)
arecord -D hw:3,0 -f S32_LE -r 705600 -c 2 -d 10 test_streaming_output.wav

# 解析
uv run python scripts/analyze_waveform.py test_streaming_output.wav
```

期待結果:
- クリック検出数: -40dBで0~数個程度に激減
- 正弦波が滑らかに再生されている

## 予想される効果

### 定量的改善
- **クリック数**: 現在の無限ループ状態から、ほぼゼロへ
- **不連続間隔**: 16,384サンプル(現在)→なし
- **ユーザー体感**: "プチプチ音"の完全消失

### 技術的改善
- Overlap-Saveアルゴリズムが正しく機能
- ブロック進行が正常化
- ステレオチャンネル間の位相連続性も保証される

## リスクと対策

### リスク1: レイテンシー増加
**内容**: 入力バッファリング(3,036サンプル蓄積待ち)により、約69msのレイテンシー追加
- 計算: 3,036 / 44,100Hz ≈ 68.8ms

**対策**:
- リスニング用途では許容範囲(元々Overlap-SaveのFFT処理で数十ms存在)
- LV2プラグイン化時にレイテンシー報告機能で対応

### リスク2: バッファオーバーフロー
**内容**: PipeWireからの入力が予想以上に速い場合、バッファ溢れの可能性

**対策**:
- `streamInputBuffer_`を2倍サイズで確保(安全マージン)
- オーバーフロー検出時は古いサンプルを破棄してログ出力

### リスク3: ステレオ同期
**内容**: 左右チャンネルで異なるタイミングで処理が発生する可能性

**対策**:
- 左右チャンネル用に独立したストリーミング状態を保持
- または、インターリーブ前の段階で同期を保証する設計

## まとめ

**現状**: `ConvolveBlock()`がオフライン処理用に設計されており、リアルタイムストリーミングでは毎回ブロックカウンタがリセットされるため、無限ループ状態になっている。

**修正方針**: 新規メソッド`ConvolveStreamBlock()`を追加し、ストリーミング用の状態保持とバッファリングを実装する。

**期待効果**: クラックリングノイズの完全消失。ブロック境界での不連続が解消され、滑らかな705.6kHz出力が実現する。

**次のステップ**: Phase 1から順に実装を進め、各フェーズでテストを実施する。
