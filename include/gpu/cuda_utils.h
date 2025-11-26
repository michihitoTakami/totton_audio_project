#ifndef GPU_CUDA_UTILS_H
#define GPU_CUDA_UTILS_H

#include "error_codes.h"

#include <cuda_runtime.h>
#include <cufft.h>

namespace ConvolutionEngine {

// RAII for short-lived host buffers; long-lived buffers are registered via
// registerHostBuffer/unregisterHostBuffers to avoid repetitive cudaHostRegister calls.
class ScopedHostPin {
   public:
    ScopedHostPin(void* ptr, size_t bytes, const char* context);
    ~ScopedHostPin();

    ScopedHostPin(const ScopedHostPin&) = delete;
    ScopedHostPin& operator=(const ScopedHostPin&) = delete;

   private:
    void* ptr_;
    bool registered_;
};

namespace Utils {

// Check CUDA errors
void checkCudaError(cudaError_t error, const char* context);

// Check cuFFT errors
void checkCufftError(cufftResult result, const char* context);

/**
 * @brief Create a CUDA stream with the highest available priority.
 *        Falls back to default priority if the device does not support priorities.
 *
 * @param context Human-readable context string for logging.
 * @param flags cudaStream flags (default: cudaStreamNonBlocking).
 * @param usedHighPriority Optional pointer that reports whether high priority was granted.
 * @return cudaStream_t Newly created CUDA stream.
 *
 * @throws std::runtime_error if stream creation fails.
 */
cudaStream_t createPrioritizedStream(const char* context,
                                     unsigned int flags = cudaStreamNonBlocking,
                                     bool* usedHighPriority = nullptr);

// Measure GPU utilization (requires nvidia-ml)
double getGPUUtilization();

// =========================================================================
// ErrorCode-based error handling (Issue #44)
// =========================================================================

/**
 * @brief Convert CUDA error to ErrorCode
 * @param error CUDA error code
 * @return Corresponding ErrorCode
 */
AudioEngine::ErrorCode cudaErrorToErrorCode(cudaError_t error);

/**
 * @brief Convert cuFFT result to ErrorCode
 * @param result cuFFT result code
 * @return GPU_CUFFT_ERROR on failure, OK on success
 */
AudioEngine::ErrorCode cufftResultToErrorCode(cufftResult result);

/**
 * @brief Check CUDA error and return ErrorCode (non-throwing)
 * @param error CUDA error code
 * @param context Context string for logging
 * @return OK on success, appropriate ErrorCode on failure
 */
AudioEngine::ErrorCode checkCudaErrorCode(cudaError_t error, const char* context = nullptr);

/**
 * @brief Check cuFFT result and return ErrorCode (non-throwing)
 * @param result cuFFT result code
 * @param context Context string for logging
 * @return OK on success, GPU_CUFFT_ERROR on failure
 */
AudioEngine::ErrorCode checkCufftErrorCode(cufftResult result, const char* context = nullptr);

}  // namespace Utils

}  // namespace ConvolutionEngine

#endif  // GPU_CUDA_UTILS_H
