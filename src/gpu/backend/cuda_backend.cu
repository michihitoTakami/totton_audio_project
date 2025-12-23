#include "gpu/backend/gpu_backend.h"

#include "logging/logger.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace ConvolutionEngine {
namespace GpuBackend {

namespace {

AudioEngine::ErrorCode mapCudaError(cudaError_t err, const char* context) {
    if (err == cudaSuccess) {
        return AudioEngine::ErrorCode::OK;
    }
    LOG_ERROR("[CUDA backend] {} failed: {}", context, cudaGetErrorString(err));
    switch (err) {
        case cudaErrorMemoryAllocation:
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        case cudaErrorInvalidDevice:
        case cudaErrorNoDevice:
            return AudioEngine::ErrorCode::GPU_DEVICE_NOT_FOUND;
        default:
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;
    }
}

AudioEngine::ErrorCode mapCufftError(cufftResult err, const char* context) {
    if (err == CUFFT_SUCCESS) {
        return AudioEngine::ErrorCode::OK;
    }
    LOG_ERROR("[CUDA backend] {} failed: {}", context, static_cast<int>(err));
    return AudioEngine::ErrorCode::GPU_CUFFT_ERROR;
}

struct CudaStreamHolder {
    cudaStream_t stream = nullptr;
};

struct CudaFftPlan {
    cufftHandle forward = 0;
    cufftHandle inverse = 0;
    FftDomain domain = FftDomain::RealToComplex;
    int fftSize = 0;
    int batch = 1;
};

struct CudaBackend final : public IGpuBackend {
    BackendKind kind() const override {
        return BackendKind::Cuda;
    }

    const char* name() const override {
        return "cuda";
    }

    AudioEngine::ErrorCode allocateDeviceBuffer(DeviceBuffer& out, size_t bytes,
                                                const char* context) override {
        void* ptr = nullptr;
        auto err = mapCudaError(cudaMalloc(&ptr, bytes), context);
        if (err != AudioEngine::ErrorCode::OK) {
            return err;
        }
        out.handle.ptr = ptr;
        out.bytes = bytes;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode freeDeviceBuffer(DeviceBuffer& buffer, const char* /*context*/) override {
        if (!buffer.handle.ptr) {
            return AudioEngine::ErrorCode::OK;
        }
        mapCudaError(cudaFree(buffer.handle.ptr), "cudaFree");
        buffer.handle.ptr = nullptr;
        buffer.bytes = 0;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode createStream(Stream& out, const char* context) override {
        auto holder = std::make_unique<CudaStreamHolder>();
        auto err = mapCudaError(cudaStreamCreateWithFlags(&holder->stream, cudaStreamNonBlocking),
                                context);
        if (err != AudioEngine::ErrorCode::OK) {
            return err;
        }
        out.handle.ptr = holder.release();
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode destroyStream(Stream& stream, const char* /*context*/) override {
        if (!stream.handle.ptr) {
            return AudioEngine::ErrorCode::OK;
        }
        auto holder = static_cast<CudaStreamHolder*>(stream.handle.ptr);
        mapCudaError(cudaStreamDestroy(holder->stream), "cudaStreamDestroy");
        delete holder;
        stream.handle.ptr = nullptr;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode streamSynchronize(const Stream* stream,
                                             const char* context) override {
        cudaStream_t s = nullptr;
        if (stream && stream->handle.ptr) {
            s = static_cast<CudaStreamHolder*>(stream->handle.ptr)->stream;
        }
        return mapCudaError(cudaStreamSynchronize(s), context);
    }

    AudioEngine::ErrorCode copy(void* dst, const void* src, size_t bytes, CopyKind kind,
                                const Stream* stream, const char* context) override {
        cudaMemcpyKind copyKind = cudaMemcpyDefault;
        switch (kind) {
            case CopyKind::HostToDevice:
                copyKind = cudaMemcpyHostToDevice;
                break;
            case CopyKind::DeviceToHost:
                copyKind = cudaMemcpyDeviceToHost;
                break;
            case CopyKind::DeviceToDevice:
                copyKind = cudaMemcpyDeviceToDevice;
                break;
        }
        cudaStream_t s = nullptr;
        if (stream && stream->handle.ptr) {
            s = static_cast<CudaStreamHolder*>(stream->handle.ptr)->stream;
        }
        auto err = mapCudaError(cudaMemcpyAsync(dst, src, bytes, copyKind, s), context);
        if (err != AudioEngine::ErrorCode::OK) {
            return err;
        }
        if (!s) {
            // cudaMemcpyAsync on default stream is effectively sync; ensure consistency.
            return AudioEngine::ErrorCode::OK;
        }
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode createFftPlan1d(FftPlan& out, int fftSize, int batch, FftDomain domain,
                                           const char* context) override {
        if (fftSize <= 0 || batch <= 0) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }
        auto plan = std::make_unique<CudaFftPlan>();
        plan->domain = domain;
        plan->fftSize = fftSize;
        plan->batch = batch;

        int n[1] = {fftSize};
        int inembed[1] = {fftSize};
        int onembedR2C[1] = {fftSize / 2 + 1};
        int onembedC2C[1] = {fftSize};

        if (domain == FftDomain::RealToComplex) {
            auto errF = mapCufftError(
                cufftPlanMany(&plan->forward, 1, n, inembed, 1, fftSize, onembedR2C, 1,
                              fftSize / 2 + 1,
                              CUFFT_R2C, batch),
                context);
            if (errF != AudioEngine::ErrorCode::OK) {
                return errF;
            }
            auto errI = mapCufftError(
                cufftPlanMany(&plan->inverse, 1, n, onembedR2C, 1, fftSize / 2 + 1, inembed, 1,
                              fftSize,
                              CUFFT_C2R, batch),
                context);
            if (errI != AudioEngine::ErrorCode::OK) {
                cufftDestroy(plan->forward);
                return errI;
            }
        } else if (domain == FftDomain::ComplexToComplex) {
            auto errF =
                mapCufftError(cufftPlanMany(&plan->forward, 1, n, inembed, 1, fftSize, onembedC2C,
                                            1, fftSize, CUFFT_C2C, batch),
                              context);
            if (errF != AudioEngine::ErrorCode::OK) {
                return errF;
            }
            auto errI =
                mapCufftError(cufftPlanMany(&plan->inverse, 1, n, inembed, 1, fftSize, onembedC2C,
                                            1, fftSize, CUFFT_C2C, batch),
                              context);
            if (errI != AudioEngine::ErrorCode::OK) {
                cufftDestroy(plan->forward);
                return errI;
            }
        } else {
            return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
        }

        plan->domain = domain;
        auto raw = plan.release();
        out.handle.ptr = raw;
        out.fftSize = fftSize;
        out.batch = batch;
        out.domain = domain;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode destroyFftPlan(FftPlan& plan, const char* /*context*/) override {
        if (!plan.handle.ptr) {
            return AudioEngine::ErrorCode::OK;
        }
        auto* raw = static_cast<CudaFftPlan*>(plan.handle.ptr);
        if (raw->forward) {
            cufftDestroy(raw->forward);
        }
        if (raw->inverse) {
            cufftDestroy(raw->inverse);
        }
        delete raw;
        plan.handle.ptr = nullptr;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode executeFft(const FftPlan& plan, const DeviceBuffer& in,
                                      DeviceBuffer& out, FftDirection direction,
                                      const Stream* stream, const char* context) override {
        if (!plan.handle.ptr || !in.handle.ptr || !out.handle.ptr) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }
        auto* rawPlan = static_cast<CudaFftPlan*>(plan.handle.ptr);
        cufftHandle handle = (direction == FftDirection::Forward) ? rawPlan->forward
                                                                  : rawPlan->inverse;
        cudaStream_t s = nullptr;
        if (stream && stream->handle.ptr) {
            s = static_cast<CudaStreamHolder*>(stream->handle.ptr)->stream;
            mapCufftError(cufftSetStream(handle, s), "cufftSetStream");
        }

        if (rawPlan->domain == FftDomain::RealToComplex) {
            if (direction == FftDirection::Forward) {
                return mapCufftError(
                    cufftExecR2C(handle, static_cast<cufftReal*>(in.handle.ptr),
                                 static_cast<cufftComplex*>(out.handle.ptr)),
                    context);
            }
            return mapCufftError(
                cufftExecC2R(handle, static_cast<cufftComplex*>(in.handle.ptr),
                             static_cast<cufftReal*>(out.handle.ptr)),
                context);
        }

        if (rawPlan->domain == FftDomain::ComplexToComplex) {
            int dir = (direction == FftDirection::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
            return mapCufftError(
                cufftExecC2C(handle, static_cast<cufftComplex*>(in.handle.ptr),
                             static_cast<cufftComplex*>(out.handle.ptr), dir),
                context);
        }

        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }

    AudioEngine::ErrorCode complexPointwiseMulScale(DeviceBuffer& out, const DeviceBuffer& a,
                                                    const DeviceBuffer& b, size_t complexCount,
                                                    float scale, const Stream* stream,
                                                    const char* context) override;
};

__global__ void complexMulScaleKernel(cufftComplex* out, const cufftComplex* a,
                                      const cufftComplex* b, size_t count, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    float ar = a[idx].x;
    float ai = a[idx].y;
    float br = b[idx].x;
    float bi = b[idx].y;
    out[idx].x = (ar * br - ai * bi) * scale;
    out[idx].y = (ar * bi + ai * br) * scale;
}

AudioEngine::ErrorCode CudaBackend::complexPointwiseMulScale(DeviceBuffer& out,
                                                             const DeviceBuffer& a,
                                                             const DeviceBuffer& b,
                                                             size_t complexCount, float scale,
                                                             const Stream* stream,
                                                             const char* context) {
    if (!out.handle.ptr || !a.handle.ptr || !b.handle.ptr) {
        return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
    }
    cudaStream_t s = nullptr;
    if (stream && stream->handle.ptr) {
        s = static_cast<CudaStreamHolder*>(stream->handle.ptr)->stream;
    }
    constexpr int kBlock = 256;
    int blocks = static_cast<int>((complexCount + kBlock - 1) / kBlock);
    complexMulScaleKernel<<<blocks, kBlock, 0, s>>>(static_cast<cufftComplex*>(out.handle.ptr),
                                                    static_cast<const cufftComplex*>(a.handle.ptr),
                                                    static_cast<const cufftComplex*>(b.handle.ptr),
                                                    complexCount, scale);
    return mapCudaError(cudaGetLastError(), context);
}

}  // namespace

std::unique_ptr<IGpuBackend> createCudaBackend() {
    return std::make_unique<CudaBackend>();
}

}  // namespace GpuBackend
}  // namespace ConvolutionEngine
