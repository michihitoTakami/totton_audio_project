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

// =========================================================================
// ErrorCode-based error handling (Issue #44)
// =========================================================================

AudioEngine::ErrorCode cudaErrorToErrorCode(cudaError_t error) {
    if (error == cudaSuccess) {
        return AudioEngine::ErrorCode::OK;
    }

    // Map specific CUDA errors to our error codes
    switch (error) {
        // Initialization errors
        case cudaErrorInitializationError:
        case cudaErrorInsufficientDriver:
        case cudaErrorIncompatibleDriverContext:
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;

        // Device errors
        case cudaErrorNoDevice:
        case cudaErrorInvalidDevice:
        case cudaErrorDeviceUninitialized:
            return AudioEngine::ErrorCode::GPU_DEVICE_NOT_FOUND;

        // Memory errors
        case cudaErrorMemoryAllocation:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidHostPointer:
        case cudaErrorHostMemoryAlreadyRegistered:
        case cudaErrorHostMemoryNotRegistered:
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;

        // Kernel launch errors
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorLaunchOutOfResources:
        case cudaErrorInvalidConfiguration:
        case cudaErrorInvalidKernelImage:
        case cudaErrorInvalidSymbol:
            return AudioEngine::ErrorCode::GPU_KERNEL_LAUNCH_FAILED;

        // Default to INTERNAL_UNKNOWN for unmapped CUDA errors
        // This makes debugging easier by distinguishing truly unknown errors
        default:
            return AudioEngine::ErrorCode::INTERNAL_UNKNOWN;
    }
}

AudioEngine::ErrorCode cufftResultToErrorCode(cufftResult result) {
    if (result == CUFFT_SUCCESS) {
        return AudioEngine::ErrorCode::OK;
    }
    return AudioEngine::ErrorCode::GPU_CUFFT_ERROR;
}

AudioEngine::ErrorCode checkCudaErrorCode(cudaError_t error, const char* context) {
    if (error == cudaSuccess) {
        return AudioEngine::ErrorCode::OK;
    }

    // Log the error
    if (context) {
        std::cerr << "CUDA Error in " << context << ": "
                  << cudaGetErrorString(error) << std::endl;
    } else {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }

    return cudaErrorToErrorCode(error);
}

AudioEngine::ErrorCode checkCufftErrorCode(cufftResult result, const char* context) {
    if (result == CUFFT_SUCCESS) {
        return AudioEngine::ErrorCode::OK;
    }

    // Log the error
    if (context) {
        std::cerr << "cuFFT Error in " << context << ": "
                  << static_cast<int>(result) << std::endl;
    } else {
        std::cerr << "cuFFT Error: " << static_cast<int>(result) << std::endl;
    }

    return AudioEngine::ErrorCode::GPU_CUFFT_ERROR;
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
