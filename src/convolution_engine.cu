#include "convolution_engine.h"
#include "filter_coefficients.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>

namespace ConvolutionEngine {

// CUDA kernel for zero-padding (insert zeros between samples for upsampling)
__global__ void zeroPadKernel(const float* input, float* output,
                              int inputLength, int upsampleRatio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputLength) {
        output[idx * upsampleRatio] = input[idx];
        // Zero out the intermediate samples
        for (int i = 1; i < upsampleRatio; ++i) {
            output[idx * upsampleRatio + i] = 0.0f;
        }
    }
}

// CUDA kernel for complex multiplication (frequency domain)
__global__ void complexMultiplyKernel(cufftComplex* data,
                                      const cufftComplex* filter,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = data[idx].x;
        float b = data[idx].y;
        float c = filter[idx].x;
        float d = filter[idx].y;

        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        data[idx].x = a * c - b * d;
        data[idx].y = a * d + b * c;
    }
}

// CUDA kernel for scaling after IFFT
__global__ void scaleKernel(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
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
    // Simplified version - returns 0 for now
    // Full implementation would require NVML library
    return 0.0;
}

} // namespace Utils

// GPUUpsampler implementation
GPUUpsampler::GPUUpsampler()
    : upsampleRatio_(1), blockSize_(8192), filterTaps_(0), fftSize_(0),
      d_filterCoeffs_(nullptr), d_filterFFT_(nullptr),
      d_inputBlock_(nullptr), d_outputBlock_(nullptr),
      d_inputFFT_(nullptr), d_convResult_(nullptr),
      fftPlanForward_(0), fftPlanInverse_(0),
      overlapSize_(0), stream_(nullptr) {
    stats_ = Stats();
}

GPUUpsampler::~GPUUpsampler() {
    cleanup();
}

bool GPUUpsampler::initialize(const std::string& filterCoeffPath,
                              int upsampleRatio,
                              int blockSize) {
    upsampleRatio_ = upsampleRatio;
    blockSize_ = blockSize;

    std::cout << "Initializing GPU Upsampler..." << std::endl;
    std::cout << "  Upsample Ratio: " << upsampleRatio_ << "x" << std::endl;
    std::cout << "  Block Size: " << blockSize_ << " samples" << std::endl;

    // Load filter coefficients
    if (!loadFilterCoefficients(filterCoeffPath)) {
        return false;
    }

    // Setup GPU resources
    if (!setupGPUResources()) {
        return false;
    }

    std::cout << "GPU Upsampler initialized successfully!" << std::endl;
    return true;
}

bool GPUUpsampler::loadFilterCoefficients(const std::string& path) {
    std::cout << "Loading filter coefficients from: " << path << std::endl;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Cannot open filter coefficients file: " << path << std::endl;
        return false;
    }

    // Get file size
    ifs.seekg(0, std::ios::end);
    size_t fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    filterTaps_ = fileSize / sizeof(float);
    std::cout << "  Filter taps: " << filterTaps_ << std::endl;

    // Read coefficients
    h_filterCoeffs_.resize(filterTaps_);
    ifs.read(reinterpret_cast<char*>(h_filterCoeffs_.data()), fileSize);

    if (!ifs) {
        std::cerr << "Error reading filter coefficients" << std::endl;
        return false;
    }

    std::cout << "  Filter loaded successfully (" << fileSize / 1024 << " KB)" << std::endl;
    return true;
}

bool GPUUpsampler::setupGPUResources() {
    try {
        std::cout << "Setting up GPU resources..." << std::endl;

        // Create CUDA stream
        Utils::checkCudaError(
            cudaStreamCreate(&stream_),
            "cudaStreamCreate"
        );

        // Calculate FFT sizes for Overlap-Save
        // For Overlap-Save: FFT size = block size + filter taps - 1
        size_t fftSizeNeeded = static_cast<size_t>(blockSize_) + static_cast<size_t>(filterTaps_) - 1;
        // Round up to next power of 2 for efficiency
        fftSize_ = 1;
        while (static_cast<size_t>(fftSize_) < fftSizeNeeded) {
            fftSize_ *= 2;
        }
        std::cout << "  FFT size: " << fftSize_ << std::endl;

        overlapSize_ = filterTaps_ - 1;
        overlapBuffer_.resize(overlapSize_, 0.0f);

        // Allocate device memory for filter coefficients
        Utils::checkCudaError(
            cudaMalloc(&d_filterCoeffs_, filterTaps_ * sizeof(float)),
            "cudaMalloc filter coefficients"
        );

        Utils::checkCudaError(
            cudaMemcpy(d_filterCoeffs_, h_filterCoeffs_.data(),
                      filterTaps_ * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy filter coefficients"
        );

        // Allocate working buffers
        size_t upsampledBlockSize = blockSize_ * upsampleRatio_;

        Utils::checkCudaError(
            cudaMalloc(&d_inputBlock_, blockSize_ * sizeof(float)),
            "cudaMalloc input block"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_outputBlock_, (upsampledBlockSize + filterTaps_) * sizeof(float)),
            "cudaMalloc output block"
        );

        // Allocate FFT buffers
        int fftComplexSize = fftSize_ / 2 + 1;

        Utils::checkCudaError(
            cudaMalloc(&d_inputFFT_, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc input FFT"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_filterFFT_, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc filter FFT"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_convResult_, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc convolution result"
        );

        // Create cuFFT plans
        Utils::checkCufftError(
            cufftPlan1d(&fftPlanForward_, fftSize_, CUFFT_R2C, 1),
            "cufftPlan1d forward"
        );

        Utils::checkCufftError(
            cufftPlan1d(&fftPlanInverse_, fftSize_, CUFFT_C2R, 1),
            "cufftPlan1d inverse"
        );

        // Pre-compute filter FFT
        float* d_filterPadded;
        Utils::checkCudaError(
            cudaMalloc(&d_filterPadded, fftSize_ * sizeof(float)),
            "cudaMalloc filter padded"
        );

        Utils::checkCudaError(
            cudaMemset(d_filterPadded, 0, fftSize_ * sizeof(float)),
            "cudaMemset filter padded"
        );

        Utils::checkCudaError(
            cudaMemcpy(d_filterPadded, d_filterCoeffs_,
                      filterTaps_ * sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpy filter to padded"
        );

        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_filterPadded, d_filterFFT_),
            "cufftExecR2C filter"
        );

        cudaFree(d_filterPadded);

        std::cout << "  GPU resources allocated successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in setupGPUResources: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

bool GPUUpsampler::processChannel(const float* inputData,
                                  size_t inputFrames,
                                  std::vector<float>& outputData) {
    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize all GPU pointers to nullptr for safe cleanup
    float* d_upsampledInput = nullptr;
    float* d_input = nullptr;
    float* d_paddedInput = nullptr;
    cufftComplex* d_inputFFT = nullptr;
    cufftComplex* d_resultFFT = nullptr;
    float* d_convResult = nullptr;

    try {
        size_t outputFrames = inputFrames * upsampleRatio_;
        outputData.resize(outputFrames, 0.0f);

        // Step 1: Zero-pad input signal (upsample)
        Utils::checkCudaError(
            cudaMalloc(&d_upsampledInput, outputFrames * sizeof(float)),
            "cudaMalloc upsampled input"
        );

        Utils::checkCudaError(
            cudaMemset(d_upsampledInput, 0, outputFrames * sizeof(float)),
            "cudaMemset upsampled input"
        );

        // Copy input to device
        Utils::checkCudaError(
            cudaMalloc(&d_input, inputFrames * sizeof(float)),
            "cudaMalloc input"
        );

        Utils::checkCudaError(
            cudaMemcpy(d_input, inputData, inputFrames * sizeof(float),
                      cudaMemcpyHostToDevice),
            "cudaMemcpy input to device"
        );

        // Launch zero-padding kernel
        int threadsPerBlock = 256;
        int blocks = (inputFrames + threadsPerBlock - 1) / threadsPerBlock;
        zeroPadKernel<<<blocks, threadsPerBlock, 0, stream_>>>(
            d_input, d_upsampledInput, inputFrames, upsampleRatio_
        );

        cudaFree(d_input);
        d_input = nullptr;

        // Step 2: Perform GPU FFT convolution using pre-computed filter FFT
        // Reuse fftPlanForward_, fftPlanInverse_, and d_filterFFT_

        // Allocate padded buffer for FFT (use pre-computed fftSize_)
        Utils::checkCudaError(
            cudaMalloc(&d_paddedInput, fftSize_ * sizeof(float)),
            "cudaMalloc padded input"
        );

        Utils::checkCudaError(
            cudaMemset(d_paddedInput, 0, fftSize_ * sizeof(float)),
            "cudaMemset padded input"
        );

        // Copy upsampled data to padded buffer
        size_t copySize = (outputFrames < static_cast<size_t>(fftSize_)) ?
                          outputFrames : static_cast<size_t>(fftSize_);
        Utils::checkCudaError(
            cudaMemcpy(d_paddedInput, d_upsampledInput,
                      copySize * sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpy to padded"
        );

        // Allocate temporary FFT buffer (reuse d_inputFFT_ for efficiency)
        int fftComplexSize = fftSize_ / 2 + 1;
        Utils::checkCudaError(
            cudaMalloc(&d_inputFFT, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc input FFT temp"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_resultFFT, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc result FFT"
        );

        // Perform forward FFT on input using pre-computed plan
        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_paddedInput, d_inputFFT),
            "cufftExecR2C input"
        );

        // Complex multiplication in frequency domain with pre-computed filter FFT
        threadsPerBlock = 256;
        blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;

        Utils::checkCudaError(
            cudaMemcpy(d_resultFFT, d_inputFFT,
                      fftComplexSize * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy FFT to result"
        );

        complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream_>>>(
            d_resultFFT, d_filterFFT_, fftComplexSize  // Use pre-computed d_filterFFT_
        );

        Utils::checkCudaError(
            cudaGetLastError(),
            "complexMultiplyKernel launch"
        );

        // Perform inverse FFT using pre-computed plan
        Utils::checkCudaError(
            cudaMalloc(&d_convResult, fftSize_ * sizeof(float)),
            "cudaMalloc conv result"
        );

        Utils::checkCufftError(
            cufftExecC2R(fftPlanInverse_, d_resultFFT, d_convResult),
            "cufftExecC2R inverse"
        );

        // Scale by FFT size (cuFFT doesn't normalize)
        float scale = 1.0f / fftSize_;
        size_t validOutputSize = (outputFrames < static_cast<size_t>(fftSize_)) ?
                                  outputFrames : static_cast<size_t>(fftSize_);
        int scaleBlocks = (validOutputSize + threadsPerBlock - 1) / threadsPerBlock;
        scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream_>>>(
            d_convResult, validOutputSize, scale
        );

        Utils::checkCudaError(
            cudaGetLastError(),
            "scaleKernel launch"
        );

        // Copy result back to host (only valid portion, already computed above)
        Utils::checkCudaError(
            cudaMemcpy(outputData.data(), d_convResult,
                      validOutputSize * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy result to host"
        );

        // If output is larger than FFT size, process in blocks
        if (outputFrames > static_cast<size_t>(fftSize_)) {
            std::cerr << "Warning: Output size (" << outputFrames
                      << ") exceeds FFT size (" << fftSize_ << ")" << std::endl;
            std::cerr << "Full Overlap-Save implementation needed for long audio." << std::endl;
            std::cerr << "Processing truncated to " << validOutputSize << " samples." << std::endl;
            outputData.resize(validOutputSize);
        }

        // Cleanup temporary buffers (reuse initialized pointers)
        cudaFree(d_upsampledInput);
        cudaFree(d_paddedInput);
        cudaFree(d_inputFFT);
        cudaFree(d_resultFFT);
        cudaFree(d_convResult);

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        stats_.totalProcessingTime += elapsed.count();
        stats_.framesProcessed += inputFrames;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in processChannel: " << e.what() << std::endl;

        // Clean up all allocated GPU memory
        if (d_upsampledInput) cudaFree(d_upsampledInput);
        if (d_input) cudaFree(d_input);
        if (d_paddedInput) cudaFree(d_paddedInput);
        if (d_inputFFT) cudaFree(d_inputFFT);
        if (d_resultFFT) cudaFree(d_resultFFT);
        if (d_convResult) cudaFree(d_convResult);

        return false;
    }
}

bool GPUUpsampler::processStereo(const float* leftInput,
                                 const float* rightInput,
                                 size_t inputFrames,
                                 std::vector<float>& leftOutput,
                                 std::vector<float>& rightOutput) {
    // Process both channels
    bool success = processChannel(leftInput, inputFrames, leftOutput);
    if (success) {
        success = processChannel(rightInput, inputFrames, rightOutput);
    }

    return success;
}

void GPUUpsampler::cleanup() {
    if (d_filterCoeffs_) cudaFree(d_filterCoeffs_);
    if (d_filterFFT_) cudaFree(d_filterFFT_);
    if (d_inputBlock_) cudaFree(d_inputBlock_);
    if (d_outputBlock_) cudaFree(d_outputBlock_);
    if (d_inputFFT_) cudaFree(d_inputFFT_);
    if (d_convResult_) cudaFree(d_convResult_);

    if (fftPlanForward_) cufftDestroy(fftPlanForward_);
    if (fftPlanInverse_) cufftDestroy(fftPlanInverse_);

    if (stream_) cudaStreamDestroy(stream_);

    d_filterCoeffs_ = nullptr;
    d_filterFFT_ = nullptr;
    d_inputBlock_ = nullptr;
    d_outputBlock_ = nullptr;
    d_inputFFT_ = nullptr;
    d_convResult_ = nullptr;
    fftPlanForward_ = 0;
    fftPlanInverse_ = 0;
    stream_ = nullptr;
}

} // namespace ConvolutionEngine
