# Vulkan Overlap-Save (Offline) - Issue #928

Vulkan compute + VkFFT を使った **1ch オフライン**のアップサンプル + FIR (Overlap-Save) 検証用サンプルです。CUDA 実装を壊さず、Vulkan バックエンドでフィルタ長 12万 tap 以上を処理できることを確認します。

## ビルド

```bash
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_VULKAN=ON \
  -DGPU_UPSAMPLER_BUILD_VULKAN_OVERLAP_SAMPLE=ON
cmake --build build-vulkan -j"$(nproc)" vulkan_overlap_save_tool vulkan_overlap_save_tests
```

## 実行

```bash
./build-vulkan/samples/vkfft_overlap_save/vulkan_overlap_save_tool \
  --input test_data/input_mono.wav \
  --output /tmp/output.wav \
  --filter data/coefficients/filter_44k_8x_2m_min_phase.bin
```

- `--filter-json` を省略すると、`--filter` の拡張子を `.json` に置き換えてメタデータを読み込みます。
- `--fft-size` を省略すると `chunkFrames` とメタデータの tap 数から自動決定します。

## テスト

```bash
ctest --output-on-failure -R VulkanOverlapSave --test-dir build-vulkan
```

`vulkan_overlap_save_tests` はインパルス応答がフィルタと一致するか確認します（環境に Vulkan デバイスがない場合は skip）。

## CUDA/Vulkan Nullテスト（同一性チェック）

- CUDA と Vulkan のストリーミングアップサンプラで同一入力を処理し、残差が十分小さいことを検証する gtest を追加しています。
- ビルド: `cmake -B build -DENABLE_VULKAN=ON -DENABLE_CUDA=ON`（または既存の build ディレクトリに同オプションを付与）
- 実行: `ctest --output-on-failure -R VulkanCudaParity --test-dir build`
- 内容: インパルス係数（256tap）、2xアップサンプル、ブロック64で処理し、CUDA/Vulkan 出力差分の RMS < 1e-3 / 最大絶対差 < 5e-3 を確認（Vulkan or CUDA が無い場合は skip）
