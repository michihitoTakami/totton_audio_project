# VkFFT 最小サンプル (Issue #1108)

Jetson Orin Nano と PC (RTX2070S) の双方で **VkFFT の 1D R2C/C2R** が実行できるか確認するための最小サンプルを追加した。`GPU_UPSAMPLER_BUILD_VKFFT_SAMPLE=ON` で有効化すると、`samples/vkfft_minimal/vkfft_minimal` がビルドされる。

## 依存関係
- Vulkan ランタイム/開発ヘッダ (`libvulkan-dev`) が導入済みであること
- 最新ドライバ (Jetson/PC ともに `vulkaninfo` が通る状態)
- CMake 3.18+ / Ninja (任意)
- glslang は FetchContent で自動取得するため追加インストール不要

Jetson/PC 共通の追加パッケージ例:
```bash
sudo apt-get install -y libvulkan-dev vulkan-validationlayers-dev
```

## ビルド手順
```bash
cmake -B build-vkfft -DCMAKE_BUILD_TYPE=Release -DGPU_UPSAMPLER_BUILD_VKFFT_SAMPLE=ON
cmake --build build-vkfft -j"$(nproc)"
```

## 実行手順
```bash
./build-vkfft/samples/vkfft_minimal/vkfft_minimal
```

出力例:
```
VkFFT 1D R2C/C2R sample
  device : NVIDIA GeForce RTX 2070 SUPER
  size   : 1024
  time   : 0.12 ms (forward+inverse)
  maxerr : 2.1e-05
  rmse   : 6.5e-06
```

- `maxerr` が `1e-3` 未満なら簡易検証をパスした扱い。
- Jetson でも同様に `maxerr` の範囲に収まることを確認する。

## メモ
- VkFFT v1.3.4 を `third_party/vkfft` に同梱。
- glslang (Apache-2.0) を FetchContent でビルド時取得。オフライン環境では事前に依存をミラーしておくこと。
- 本サンプルは Vulkan バックエンド導入検証用であり、既存 CUDA パスには影響しない。
