#ifndef GPU_CUDA_UTILS_H
#define GPU_CUDA_UTILS_H

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

// Measure GPU utilization (requires nvidia-ml)
double getGPUUtilization();

}  // namespace Utils

}  // namespace ConvolutionEngine

#endif  // GPU_CUDA_UTILS_H
