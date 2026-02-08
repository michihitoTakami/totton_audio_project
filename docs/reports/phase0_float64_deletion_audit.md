# Phase 0: float64削除対象の棚卸しと回帰基準確定

**Issue:** #1301
**親EPIC:** #1300 - GPU upsampler float64経路の削除とfloat32単一化
**作成日:** 2026-02-08

## 目的

削除作業の前提として、float64依存箇所を確定し、回帰確認の最小セットを固定する。

## 1. `GPU_UPSAMPLER_USE_FLOAT64` 参照箇所の棚卸し

### 1.1 ビルドシステム

#### CMakeLists.txt
- **L81**: `option(GPU_UPSAMPLER_USE_FLOAT64 "Build GPU upsampler with float64 pipeline" OFF)`
  - CMakeオプション定義（削除対象）
- **L91-96**: 条件付きコンパイル定義
  ```cmake
  if(GPU_UPSAMPLER_USE_FLOAT64)
      add_compile_definitions(GPU_UPSAMPLER_USE_FLOAT64)
      message(STATUS "GPU upsampler precision: float64 (double)")
  else()
      message(STATUS "GPU upsampler precision: float32")
  endif()
  ```
  - 削除対象: `if`ブロック全体
  - 残存: float32のmessageのみ

### 1.2 コアヘッダー

#### include/gpu/precision_traits.h
- **L41-62**: `PrecisionTraits<double>` 特殊化
  - 削除対象: `template <> struct PrecisionTraits<double>` 定義全体
  - 影響: double専用のFFT型 (CUFFT_D2Z/CUFFT_Z2D)、cufftDoubleComplex
- **L64-68**: `ActivePrecision` 型エイリアス選択
  ```cpp
  #ifdef GPU_UPSAMPLER_USE_FLOAT64
  using ActivePrecision = double;
  #else
  using ActivePrecision = float;
  #endif
  ```
  - 削除対象: `#ifdef`ブロック全体
  - 置き換え: `using ActivePrecision = float;` に単純化
- **L79-87**: `copyHostToDeviceSamples` のdouble分岐
  ```cpp
  if constexpr (Traits::kIsDouble) {
      std::vector<typename Traits::Sample> temp(count);
      for (std::size_t i = 0; i < count; ++i) {
          temp[i] = static_cast<typename Traits::Sample>(src[i]);
      }
      return cudaMemcpy(dst, temp.data(), count * sizeof(typename Traits::Sample),
                        cudaMemcpyHostToDevice);
  }
  return cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
  ```
  - 削除対象: `if constexpr` ブロック
  - 簡略化: 直接 `cudaMemcpy` のみ残す
- **L93-106**: `copyDeviceToHostSamples` のdouble分岐
  - 同様に簡略化
- **L117-124**: `convertCoefficientsToPrecision` のコメント
  - コメント削除（float32のみになるため自明）

### 1.3 CUDA実装ファイル

#### src/gpu/four_channel_fir.cu
- **L197-202**: float64用scratch buffer確保
  ```cpp
  #if defined(GPU_UPSAMPLER_USE_FLOAT64)
      Utils::checkCudaError(
          cudaMalloc(&d_inputScratchL_, streamValidInputPerBlock_ * sizeof(float)),
          "cudaMalloc input scratch L");
      Utils::checkCudaError(
          cudaMalloc(&d_inputScratchR_, streamValidInputPerBlock_ * sizeof(float)),
          "cudaMalloc input scratch R");
  #endif
  ```
  - 削除対象: `#if defined` ブロック全体
  - 削除対象: `d_inputScratchL_`, `d_inputScratchR_` メンバ変数
- **L301-306**: `copyHostFloatToDeviceSamplesAsync` 内のdouble変換処理
  - 削除対象: `#if defined` ブロック
  - 簡略化: 直接コピー経路のみ残す
- **L736-739**: デストラクタでの条件付きメモリ解放
  ```cpp
  #if defined(GPU_UPSAMPLER_USE_FLOAT64)
      freeIf(d_inputScratchL_);
      freeIf(d_inputScratchR_);
  #endif
  ```
  - 削除対象: `#if defined` ブロック全体

#### src/gpu/gpu_upsampler_core.cu
- **L25, L52, L97, L115, L144, L403**: `if constexpr (Precision::kIsDouble)` 分岐
  - すべて削除対象
  - 各箇所でfloat32経路のみ残す
  - `d_outputScratch_` バッファ確保・使用ロジックも削除
- 影響:
  - `copyDeviceToHostSamplesAsync` 関数の簡略化
  - `copyHostToDeviceSamplesAsync` 関数の簡略化
  - `getOutputScratch` 関数の削除

#### src/gpu/gpu_upsampler_streaming.cu
- `ActivePrecision` 型エイリアスを使用（間接的依存）
- float32単一化後も `ActivePrecision = float` として継続使用
- **変更不要**: 型エイリアス経由のため、定義変更のみで対応

#### src/gpu/gpu_upsampler_multi_rate.cu
- 同上
- **変更不要**

### 1.4 テストコード

#### tests/gpu/test_convolution_engine.cu
- **L222-229**: `Float64BuildLoadsFloatCoefficients` テスト
  ```cpp
  TEST_F(ConvolutionEngineTest, Float64BuildLoadsFloatCoefficients) {
      if (!ActivePrecisionTraits::kIsDouble) {
          GTEST_SKIP() << "Requires GPU_UPSAMPLER_USE_FLOAT64 build";
      }
      // ...
  }
  ```
  - **削除対象**: テストケース全体
  - 理由: float64専用テストケース

#### tests/cpp/core/test_precision_traits.cpp
- **L42-46**: ActivePrecision選択のテスト
  ```cpp
  TEST(PrecisionTraitsTest, ActivePrecisionSelection) {
  #ifdef GPU_UPSAMPLER_USE_FLOAT64
      constexpr bool expectDouble = true;
  #else
      constexpr bool expectDouble = false;
  #endif
      constexpr bool actualIsDouble = ActivePrecisionTraits::kIsDouble;
      // ...
  }
  ```
  - **削除対象**: `#ifdef` ブロック
  - **置き換え**: `constexpr bool expectDouble = false;` に固定
  - または、テストケース全体を簡略化（float32前提の検証のみ）

- **L30-39**: `PrecisionTraitsTest.DoubleTraits` テスト
  - **削除対象**: テストケース全体
  - 理由: double専用トレイトテスト

### 1.5 ドキュメント

#### docs/setup/build.md
- **L92-102**: float64ビルドオプションの説明
  ```markdown
  ### float64ビルド

  GPUパイプラインをdouble精度でビルドするオプション。...

  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_UPSAMPLER_USE_FLOAT64=ON
  cmake --build build -j$(nproc)
  # ...
  ```
  ```
  - **削除対象**: セクション全体

## 2. 実行すべき回帰テスト（GPU/RT経路）

### 2.1 必須回帰テストセット

float64削除後、以下のテストが **float32ビルドで** 正常に通過することを確認する。

#### GPU統合テスト
- **ターゲット**: `gpu_tests`
  - **ファイル**: `tests/gpu/test_convolution_engine.cu`
  - **テストケース**:
    - `ConvolutionEngineTest.BasicConvolution`
    - `ConvolutionEngineTest.MultiChannelConvolution`
    - `ConvolutionEngineTest.StreamingConvolution`
    - ~~`ConvolutionEngineTest.Float64BuildLoadsFloatCoefficients`~~ (削除)
  - **実行コマンド**:
    ```bash
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target gpu_tests -j$(nproc)
    ./build/gpu_tests
    ```

- **ターゲット**: `gpu_backend_cuda_tests`
  - **ファイル**: `tests/cpp/gpu/backend/test_cuda_backend.cpp`
  - **実行コマンド**:
    ```bash
    ./build/gpu_backend_cuda_tests
    ```

#### 4チャンネルFIRテスト
- **ターゲット**: `gpu_tests` (four_channel_fir含む)
  - **ファイル**: `tests/gpu/test_four_channel_fir.cu`
  - **テストケース**:
    - 全テストケースが対象
  - **備考**: `d_inputScratchL_/R_` 削除の影響確認

#### Precision Traits テスト
- **ターゲット**: `cpu_tests` (precision_traits含む)
  - **ファイル**: `tests/cpp/core/test_precision_traits.cpp`
  - **テストケース**:
    - `PrecisionTraitsTest.FloatTraits`
    - ~~`PrecisionTraitsTest.DoubleTraits`~~ (削除)
    - `PrecisionTraitsTest.ActivePrecisionSelection` (簡略化)
  - **実行コマンド**:
    ```bash
    ./build/cpu_tests --gtest_filter=PrecisionTraitsTest.*
    ```

### 2.2 E2E回帰テスト（推奨）

実環境相当のエンドツーエンドテストを実施し、音質・レイテンシに変化がないことを確認。

#### Audio Pipeline テスト
- **ターゲット**: `auto_negotiation_tests`, `rate_switch_e2e_tests`, `soft_mute_tests`
- **実行コマンド**:
  ```bash
  ./build/auto_negotiation_tests
  ./build/rate_switch_e2e_tests
  ./build/soft_mute_tests
  ```

#### GPU処理性能確認
- **方法**: ベンチマークまたはリアルタイム処理ログで確認
  - PCでの処理倍率（~28x realtime想定）
  - VRAM使用量（float32専用になり約半減を期待）
- **実行コマンド**:
  ```bash
  # デーモン起動後、ログで処理倍率を確認
  ./scripts/daemon.sh start
  # ステータス確認でVRAM使用量をモニタ
  ```

### 2.3 CI統合

- **CI設定**: `.github/workflows/` 配下のワークフロー
- **確認項目**:
  - float32ビルドのみがCIで実行されること
  - ~~float64ビルドジョブが削除されていること~~ (存在する場合)

## 3. 既存Issue (#775/#783/#839) との関係整理

### 3.1 Issue #775: Float64 GPU Upsampler Option (RFC)

**タイトル**: [Epic]Float64 GPU Upsampler Option (RFC)
**状態**: OPEN
**関係**: **Phase 0-3完了時にCLOSE**

#### 概要
- float64オプションの導入RFC（要求仕様書）
- サブイシュー #780, #781, #782, #783 を持つ

#### Close条件
- [x] float64実装が完了している（既存）
- [ ] **EPIC #1300の全フェーズ（#1301-#1304）が完了**
  - Phase 0 (#1301): 削除対象の棚卸し完了
  - Phase 1 (#1302): CMake/precision traits からfloat64経路を削除
  - Phase 2 (#1303): テスト/CI/Docsのfloat64依存を除去
  - Phase 3 (#1304): 最終回帰確認
- [ ] 削除理由を明記したコメントを残して CLOSE
  - コメント例: 「float64経路は実運用で活用されず保守コスト増のため、EPIC #1300にて削除完了。RFC当初の目的は達成されたが、運用判断により撤回。」

#### 関連PRの扱い
- サブイシュー #780-#783 のPRは **マージ済み** の場合、revert不要
- 削除対象コードは Phase 1-2 で段階的に削除

---

### 3.2 Issue #783: Float64対応: テスト・ベンチ計測追加

**タイトル**: Float64対応: テスト・ベンチ計測追加
**状態**: OPEN
**関係**: **Phase 2完了時にCLOSE**

#### 概要
- float64ビルドでの回帰防止テスト・ベンチマーク追加
- float32/float64双方で基本処理が通ることを確認

#### Close条件
- [x] float64対応テストが追加済み（既存）
- [ ] **Phase 2 (#1303) にて float64専用テストを削除**
  - `ConvolutionEngineTest.Float64BuildLoadsFloatCoefficients` 削除
  - `PrecisionTraitsTest.DoubleTraits` 削除
  - `PrecisionTraitsTest.ActivePrecisionSelection` 簡略化
- [ ] float32テストのみが残り、回帰が確認できる状態
- [ ] Issue CLOSEコメント例: 「float64テスト追加の目的は達成済み。EPIC #1300によりfloat64経路削除のため、該当テストも削除しCLOSE。float32回帰テストは引き続き維持。」

---

### 3.3 Issue #839: 64bitビルドでアップサンプラー出力が極小(-76dBFS)になり最終出力ほぼ無音

**タイトル**: 64bitビルドでアップサンプラー出力が極小(-76dBFS)になり最終出力ほぼ無音
**状態**: OPEN
**関係**: **Phase 0で関係性明記、Phase 3完了後にCLOSE**

#### 概要
- 64bitコンパイル時のPCM変換スケール不整合によるバグ
- GPU float64 (`GPU_UPSAMPLER_USE_FLOAT64`) とは **直接関係なし**
  - **64bit CPU アーキテクチャ** (long/size_t が64bit) の問題
  - GPU精度設定とは独立

#### Close条件（選択肢）

**Option A: 修正せずにCLOSE（推奨）**
- [ ] **根本原因**: 64bitビルド特有のスケール定数・型サイズ依存
- [ ] **判断**: float32/64削除とは無関係、64bitビルド自体が非サポート
- [ ] **Phase 3完了後にCLOSE**
  - コメント例: 「本Issueは64bit CPUビルド時の問題であり、GPU float64オプションとは独立。現在は32bitビルドが標準であり、64bitビルドは非サポートのため、修正せずCLOSE。将来64bitサポートが必要になった場合は再オープンを検討。」

**Option B: 修正してからCLOSE**
- [ ] 64bitビルドのスケール定数・型依存を修正（PCM変換部分）
- [ ] 64bitビルドでの回帰テスト追加
- [ ] **Phase 3以降で個別に対応**（EPIC #1300とは別トラック）

#### 推奨アクション
- **Option A** を採用
- 理由:
  - 64bitビルドは実運用で使用されていない
  - GPU float64削除とは別の問題領域
  - 修正コストに対する効果が低い

---

## 4. 削除対象ファイル/分岐の一覧（Summary）

### 4.1 完全削除対象

| ファイル | 削除対象 |
|---------|---------|
| `tests/gpu/test_convolution_engine.cu` | `Float64BuildLoadsFloatCoefficients` テストケース |
| `tests/cpp/core/test_precision_traits.cpp` | `DoubleTraits` テストケース |

### 4.2 部分削除・簡略化対象

| ファイル | 削除/簡略化箇所 | 残存ロジック |
|---------|----------------|-------------|
| `CMakeLists.txt` | `option(GPU_UPSAMPLER_USE_FLOAT64 ...)` および条件分岐 | float32メッセージのみ |
| `include/gpu/precision_traits.h` | `PrecisionTraits<double>` 特殊化、`#ifdef GPU_UPSAMPLER_USE_FLOAT64` | `PrecisionTraits<float>` のみ、`using ActivePrecision = float;` |
| `src/gpu/four_channel_fir.cu` | `#if defined(GPU_UPSAMPLER_USE_FLOAT64)` ブロック3箇所、scratch バッファ | 直接コピー経路 |
| `src/gpu/gpu_upsampler_core.cu` | `if constexpr (Precision::kIsDouble)` ブロック6箇所、`d_outputScratch_` | float32経路 |
| `tests/cpp/core/test_precision_traits.cpp` | `#ifdef GPU_UPSAMPLER_USE_FLOAT64` in `ActivePrecisionSelection` | `expectDouble = false;` 固定 |
| `docs/setup/build.md` | float64ビルドセクション | float32ビルド説明のみ |

### 4.3 変更不要（型エイリアス経由で自動対応）

| ファイル | 理由 |
|---------|------|
| `src/gpu/gpu_upsampler_streaming.cu` | `ActivePrecision` 経由、定義変更のみで対応 |
| `src/gpu/gpu_upsampler_multi_rate.cu` | 同上 |

---

## 5. Phase 1以降での作業指針

### Phase 1 (#1302): CMake/precision traits からfloat64経路を削除
- **対象**:
  - CMakeLists.txt: オプション削除
  - precision_traits.h: `PrecisionTraits<double>` 削除、`ActivePrecision = float` 固定
  - gpu_upsampler_core.cu: `if constexpr` 分岐削除
  - four_channel_fir.cu: `#if defined` 分岐削除

### Phase 2 (#1303): テスト/CI/Docsのfloat64依存を除去
- **対象**:
  - test_convolution_engine.cu: Float64テストケース削除
  - test_precision_traits.cpp: Doubleテストケース削除、ActivePrecision簡略化
  - build.md: float64セクション削除

### Phase 3 (#1304): 最終回帰確認と関連Issue整理
- **対象**:
  - 回帰テスト実行（Section 2参照）
  - Issue #775, #783, #839 のCLOSE処理
  - EPIC #1300 の完了確認

---

## 6. まとめ

### 完了条件（Phase 0）

- [x] `GPU_UPSAMPLER_USE_FLOAT64` 参照箇所の棚卸し完了
- [x] 削除対象ファイル/分岐の一覧作成
- [x] 実行すべき回帰テスト（GPU/RT経路）を定義
- [x] 既存Issue (#775/#783/#839) との関係整理（close条件を明記）

このドキュメントは **Phase 1-3の実装ガイド** として機能し、PRで参照可能な状態となる。

---

**次のステップ**: Phase 1 (#1302) にて、CMakeLists.txt と precision_traits.h からfloat64経路を削除開始。
