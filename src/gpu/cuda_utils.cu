#include "gpu/cuda_utils.h"
#include <iostream>

#ifdef HAVE_NVML
#include <nvml.h>
#endif

namespace ConvolutionEngine {

// ScopedHostPin implementation
ScopedHostPin::ScopedHostPin(void* ptr, size_t bytes, const char* context)
    : ptr_(ptr), registered_(false) {
    if (ptr_ && bytes > 0) {
        Utils::checkCudaError(
            cudaHostRegister(ptr_, bytes, cudaHostRegisterDefault),
            context
        );
        registered_ = true;
    }
}

ScopedHostPin::~ScopedHostPin() {
    if (registered_) {
        cudaHostUnregister(ptr_);
    }
}

// Utility functions implementation
namespace Utils {

void checkCudaError(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error in " << context << ": "
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void checkCufftError(cufftResult result, const char* context) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT Error in " << context << ": "
                  << static_cast<int>(result) << std::endl;
        throw std::runtime_error("cuFFT error");
    }
}

double getGPUUtilization() {
#ifdef HAVE_NVML
    static bool nvmlInitialized = false;
    static nvmlDevice_t device;

    // Initialize NVML on first call
    if (!nvmlInitialized) {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "Warning: Failed to initialize NVML: "
                      << nvmlErrorString(result) << std::endl;
            return -1.0;
        }

        // Get device handle for GPU 0
        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Warning: Failed to get NVML device handle: "
                      << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return -1.0;
        }

        nvmlInitialized = true;
    }

    // Query GPU utilization
    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        std::cerr << "Warning: Failed to query GPU utilization: "
                  << nvmlErrorString(result) << std::endl;
        return -1.0;
    }

    return static_cast<double>(utilization.gpu);
#else
    // NVML not available - return -1 to indicate unavailable
    return -1.0;
#endif
}

}  // namespace Utils

}  // namespace ConvolutionEngine
