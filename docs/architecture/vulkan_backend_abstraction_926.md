## Vulkan移植: CUDA依存点の最小境界（Backend抽象）を定義する（Issue #926）

### 目的
CUDA前提で実装されているアップサンプラー/畳み込み（`src/gpu/*.cu`）を、まずは**最適化は後回し**で Vulkan compute に「単純置換」できるように、置換点（依存点）を**最小の境界**として切り出す。

### 現状のCUDA依存点（依存の種類）
実装上の依存は大きく以下に分類できる：

- **メモリ**: `cudaMalloc/cudaFree`、（RT用）`cudaHostRegister/cudaHostUnregister`、（一部）一時 `cudaMalloc`（非RT）
- **コピー**: `cudaMemcpy/cudaMemcpyAsync`（H2D/D2H/D2D）
- **同期/キュー**: `cudaStream_t`、`cudaStreamSynchronize`、（RTパイプライン）`cudaEvent*` と `cudaEventQuery`
- **FFT**: `cufftHandle`、`cufftPlan1d`、`cufftExec*`、`cufftSetStream`
- **カーネル**: 複素数pointwise乗算・スケール、ゼロ埋め、型変換（float32<->float64）、オーバーラップ処理 等

主な該当ファイル（例）:

- `src/gpu/gpu_upsampler_core.cu`
- `src/gpu/gpu_upsampler_streaming.cu`
- `src/gpu/gpu_upsampler_eq.cu`
- `src/gpu/gpu_upsampler_multi_rate.cu`
- `src/gpu/four_channel_fir.cu`
- `include/gpu/cuda_utils.h` / `src/gpu/cuda_utils.cu`

### 最小境界（必須スコープ）
Issue #926 のスコープに合わせ、Vulkanバックエンド実装で「まず必要になる責務」を次に限定する：

- **バッファ確保/解放**
  - Device local 相当（CUDAのdevice memory）
  - サイズ管理（bytes）
- **H2D/D2H 相当のコピー**
  - 同期/非同期の両方（ただし初期は “非同期APIでも内部同期” でも可）
  - ストリーム/キュー指定（nullable）
- **1D FFT**
  - `R2C`（forward）と `C2R`（inverse）
  - バッチ実行（将来のパーティション/複数チャネルのため）
  - ストリーム/キュー指定（nullable）
- **複素数pointwise 乗算 + スケール**
  - `out[i] = a[i] * b[i] * scale`
  - （注意）C2R 後の 1/N スケーリングをどこでやるかは backend 側で吸収して良い
- **同期**
  - 最低限 `stream/queue synchronize` があればよい

### 任意（後続Issueへ回すもの）
EPIC #921 の「まず動く」段階では不要、また Issue #926 の任意スコープと整合するため、以下は**後続Issue**へ送る：

- **ストリーミング/イベントパイプライン**
  - CUDA: `cudaEventRecord/cudaEventQuery` で in-flight 管理
  - Vulkan: timeline semaphore / fence に置換するが、初期は “同期実行” で良い
- **最適化**
  - GPU常駐リング、転送削減、カーネル融合、同期削減

### 抽象の形（提案）
**backend層**を `CudaBackend` / `VulkanBackend` に分け、上位（畳み込み/アップサンプルのアルゴリズム）は backend に依存する。

- **Backend API案**: `include/gpu/backend/gpu_backend.h`
  - CUDA/Vulkan 型をヘッダに漏らさないため、opaque handle で統一
  - 必須スコープ（alloc/copy/fft/mul+scale/sync）だけをまず定義
  - event は optional として stub（NOT_IMPLEMENTED）で置く

### CUDA版 / Vulkan版の責務分離（見通し）
単純置換の最短経路は以下：

- **CUDA実装（既存）**は「現状の `cudaMalloc/cufftExec/...` を backend 実装に移す」だけに寄せる
- **Vulkan実装（新規）**は同じ backend API を VkFFT（FFT）+ compute kernel（mul/scale 等）で実装する
- 上位の `GPUUpsampler` / `FourChannelFIR` は **「FFT + pointwise + copy」** を backend 呼び出しに置換することで、最小変更で差し替え可能になる

### 次の一手（EPIC #921 への接続）
- FFTはまず VkFFT を採用（Issue #923）。
- その後、`VulkanBackend` の最小実装（alloc/copy/fft/mul+scale/sync）を作り、**オフライン 1ch overlap-save** で動作確認へ進める。

### 実装状況（#1169）
- `gpu_backend` に Vulkan 実装を追加（VkFFT + compute シェーダで複素乗算/スケール）。
- 1D R2C/C2R の往復と複素乗算スケールをカバーする単体テストを追加。
- メモリは初期版としてホスト可視メモリ＋同期実行で実装（性能最適化と非同期化は後続課題）。
