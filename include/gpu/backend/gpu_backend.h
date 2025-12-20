// Vulkan移植に向けた GPU Backend 抽象（設計用ヘッダ）
//
// NOTE:
// - 現時点では既存のCUDA実装を置換しません（Issue #926 の「最小境界の定義」をコード化する目的）。
// - CUDA/Vulkan の型（cudaStream_t / VkQueue 等）をヘッダに漏らさないため、すべて opaque handle
// にしています。
// - 実装は将来 `src/gpu/backend/{cuda,vulkan}_backend.*` として追加する想定です。
//
// スコープ（Issue #926 必須）:
// - バッファ確保/解放
// - H2D / D2H 相当（同期/非同期）
// - 1D FFT（R2C / C2R）、forward/inverse
// - 複素数 pointwise 乗算（+スケール）
// - 同期
//
// 任意（後続Issue）:
// - ストリーミング/イベントによるパイプライン（timeline semaphore / cudaEvent）
// - カーネル融合・最適化、常駐リング、メモリ転送削減

#ifndef GPU_BACKEND_GPU_BACKEND_H
#define GPU_BACKEND_GPU_BACKEND_H

#include "core/error_codes.h"

#include <cstddef>
#include <cstdint>

namespace ConvolutionEngine {
namespace GpuBackend {

enum class BackendKind : uint8_t {
    Cuda = 0,
    Vulkan = 1,
};

// Opaque handles to avoid leaking backend-specific types.
// - CUDA: handle may point to cudaStream_t/cufftHandle wrappers.
// - Vulkan: handle may store VkDevice/VkQueue/VkBuffer/VkFence/VkSemaphore wrappers.
struct OpaqueHandle {
    void* ptr = nullptr;
};

struct DeviceBuffer {
    OpaqueHandle handle{};
    size_t bytes = 0;
};

struct Stream {
    OpaqueHandle handle{};
};

struct Event {
    OpaqueHandle handle{};
};

enum class CopyKind : uint8_t {
    HostToDevice = 0,
    DeviceToHost = 1,
    DeviceToDevice = 2,
};

enum class FftDomain : uint8_t {
    RealToComplex = 0,
    ComplexToReal = 1,
    ComplexToComplex = 2,
};

enum class FftDirection : uint8_t {
    Forward = 0,
    Inverse = 1,
};

// FFT plan handle. 実装は VkFFT / cuFFT 等の plan を保持する。
struct FftPlan {
    OpaqueHandle handle{};
    int fftSize = 0;  // N
    int batch = 1;
    FftDomain domain = FftDomain::RealToComplex;
};

// Backend interface for minimal Vulkan port.
class IGpuBackend {
   public:
    virtual ~IGpuBackend() = default;

    virtual BackendKind kind() const = 0;
    virtual const char* name() const = 0;

    // -------- Memory / buffers --------
    virtual AudioEngine::ErrorCode allocateDeviceBuffer(DeviceBuffer& out, size_t bytes,
                                                        const char* context) = 0;
    virtual AudioEngine::ErrorCode freeDeviceBuffer(DeviceBuffer& buffer, const char* context) = 0;

    // -------- Streams / sync --------
    // stream == nullptr の場合はデフォルトキュー/同期実行で良い（最適化は後回し）。
    virtual AudioEngine::ErrorCode createStream(Stream& out, const char* context) = 0;
    virtual AudioEngine::ErrorCode destroyStream(Stream& stream, const char* context) = 0;
    virtual AudioEngine::ErrorCode streamSynchronize(const Stream* stream, const char* context) = 0;

    // -------- Copies --------
    // 非同期コピーを許容。stream == nullptr の場合は同期でも良い。
    virtual AudioEngine::ErrorCode copy(void* dst, const void* src, size_t bytes, CopyKind kind,
                                        const Stream* stream, const char* context) = 0;

    // -------- FFT --------
    // 1D FFT (R2C/C2R/C2C) の plan を作り、execute で実行する。
    virtual AudioEngine::ErrorCode createFftPlan1d(FftPlan& out, int fftSize, int batch,
                                                   FftDomain domain, const char* context) = 0;
    virtual AudioEngine::ErrorCode destroyFftPlan(FftPlan& plan, const char* context) = 0;
    virtual AudioEngine::ErrorCode executeFft(const FftPlan& plan, const DeviceBuffer& in,
                                              DeviceBuffer& out, FftDirection direction,
                                              const Stream* stream, const char* context) = 0;

    // -------- Pointwise complex operations --------
    // out[i] = a[i] * b[i] * scale  （複素数、要素数 = complexCount）
    // ※ complexCount は "N/2+1" 等のR2Cサイズを想定。
    virtual AudioEngine::ErrorCode complexPointwiseMulScale(
        DeviceBuffer& out, const DeviceBuffer& a, const DeviceBuffer& b, size_t complexCount,
        float scale, const Stream* stream, const char* context) = 0;

    // -------- Optional (future) --------
    // timeline semaphore / cudaEvent 相当。Issue #926 の必須ではないので、当面は未実装でもOK。
    virtual AudioEngine::ErrorCode createEvent(Event& out, const char* context) {
        (void)out;
        (void)context;
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
    virtual AudioEngine::ErrorCode destroyEvent(Event& event, const char* context) {
        (void)event;
        (void)context;
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
    virtual AudioEngine::ErrorCode recordEvent(Event& event, const Stream* stream,
                                               const char* context) {
        (void)event;
        (void)stream;
        (void)context;
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
    virtual AudioEngine::ErrorCode queryEvent(const Event& event, bool& outReady,
                                              const char* context) {
        (void)event;
        (void)outReady;
        (void)context;
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
};

}  // namespace GpuBackend
}  // namespace ConvolutionEngine

#endif  // GPU_BACKEND_GPU_BACKEND_H
