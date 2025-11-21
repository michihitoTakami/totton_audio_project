#include "convolution_engine.h"
#include "filter_coefficients.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <nvml.h>

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
    static bool nvmlInitialized = false;
    static nvmlDevice_t device;

    // Initialize NVML on first call
    if (!nvmlInitialized) {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "Warning: Failed to initialize NVML: "
                      << nvmlErrorString(result) << std::endl;
            return 0.0;
        }

        // Get device handle for GPU 0
        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Warning: Failed to get NVML device handle: "
                      << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 0.0;
        }

        nvmlInitialized = true;
    }

    // Query GPU utilization
    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        std::cerr << "Warning: Failed to query GPU utilization: "
                  << nvmlErrorString(result) << std::endl;
        return 0.0;
    }

    return static_cast<double>(utilization.gpu);
}

} // namespace Utils

// GPUUpsampler implementation
GPUUpsampler::GPUUpsampler()
    : upsampleRatio_(1), blockSize_(8192), filterTaps_(0), fftSize_(0),
      d_filterCoeffs_(nullptr), d_filterFFT_(nullptr),
      d_inputBlock_(nullptr), d_outputBlock_(nullptr),
      d_inputFFT_(nullptr), d_convResult_(nullptr),
      fftPlanForward_(0), fftPlanInverse_(0),
      overlapSize_(0), stream_(nullptr), streamLeft_(nullptr), streamRight_(nullptr),
      streamValidInputPerBlock_(0), streamInitialized_(false), validOutputPerBlock_(0),
      d_streamInput_(nullptr), d_streamUpsampled_(nullptr), d_streamPadded_(nullptr),
      d_streamInputFFT_(nullptr), d_streamConvResult_(nullptr) {
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

        // Create CUDA streams (one for mono, two for stereo parallel processing)
        Utils::checkCudaError(
            cudaStreamCreate(&stream_),
            "cudaStreamCreate primary"
        );

        Utils::checkCudaError(
            cudaStreamCreate(&streamLeft_),
            "cudaStreamCreate left"
        );

        Utils::checkCudaError(
            cudaStreamCreate(&streamRight_),
            "cudaStreamCreate right"
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
        overlapBufferRight_.resize(overlapSize_, 0.0f);

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

    bool success = processChannelWithStream(inputData, inputFrames, outputData,
                                            stream_, overlapBuffer_);

    if (success) {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        stats_.totalProcessingTime += elapsed.count();
        stats_.framesProcessed += inputFrames;
        stats_.gpuUtilization = Utils::getGPUUtilization();
    }

    return success;
}

bool GPUUpsampler::processChannelWithStream(const float* inputData,
                                            size_t inputFrames,
                                            std::vector<float>& outputData,
                                            cudaStream_t stream,
                                            std::vector<float>& overlapBuffer) {
    // Initialize all GPU pointers to nullptr for safe cleanup
    float* d_upsampledInput = nullptr;
    float* d_input = nullptr;
    float* d_paddedInput = nullptr;
    cufftComplex* d_inputFFT = nullptr;
    float* d_convResult = nullptr;

    try {
        size_t outputFrames = inputFrames * upsampleRatio_;
        outputData.resize(outputFrames, 0.0f);

        // Step 1: Zero-pad input signal (upsample) in one go
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
            cudaMemcpyAsync(d_input, inputData, inputFrames * sizeof(float),
                           cudaMemcpyHostToDevice, stream),
            "cudaMemcpy input to device"
        );

        // Launch zero-padding kernel
        int threadsPerBlock = 256;
        int blocks = (inputFrames + threadsPerBlock - 1) / threadsPerBlock;
        zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            d_input, d_upsampledInput, inputFrames, upsampleRatio_
        );

        cudaFree(d_input);
        d_input = nullptr;

        // Step 2: Perform Overlap-Save FFT convolution in blocks
        // Allocate persistent buffers for block processing
        Utils::checkCudaError(
            cudaMalloc(&d_paddedInput, fftSize_ * sizeof(float)),
            "cudaMalloc padded input"
        );

        int fftComplexSize = fftSize_ / 2 + 1;
        Utils::checkCudaError(
            cudaMalloc(&d_inputFFT, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc input FFT"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_convResult, fftSize_ * sizeof(float)),
            "cudaMalloc conv result"
        );

        // Overlap-Save parameters
        // validOutputPerBlock is the number of new output samples per block
        // (excluding the overlap region that gets discarded)
        int validOutputPerBlock = fftSize_ - overlapSize_;

        // NOTE: Do NOT reset overlap buffer here - it must persist between stereo channels
        // to maintain phase continuity. Buffer is initialized once in setupGPUResources().
        // CRITICAL FIX: Removing this line resolves crackling noise in stereo processing.
        // std::fill(overlapBuffer.begin(), overlapBuffer.end(), 0.0f);

        // Process audio in blocks
        size_t outputPos = 0;
        size_t inputPos = 0;
        size_t blockCount = 0;  // DEBUG: Track block number

        while (outputPos < outputFrames) {
            // Calculate current block size
            size_t remainingSamples = outputFrames - inputPos;
            size_t currentBlockSize = (remainingSamples < static_cast<size_t>(validOutputPerBlock)) ?
                                       remainingSamples : static_cast<size_t>(validOutputPerBlock);

            // Prepare input block: [overlap from previous | new input data]
            Utils::checkCudaError(
                cudaMemsetAsync(d_paddedInput, 0, fftSize_ * sizeof(float), stream),
                "cudaMemset padded input"
            );

            // Copy overlap from previous block (host to device)
            if (overlapSize_ > 0 && outputPos > 0) {
                Utils::checkCudaError(
                    cudaMemcpyAsync(d_paddedInput, overlapBuffer.data(),
                                   overlapSize_ * sizeof(float), cudaMemcpyHostToDevice, stream),
                    "cudaMemcpy overlap to device"
                );

                // DEBUG: Log overlap buffer state (first few blocks only to avoid spam)
                if (blockCount < 3) {
                    fprintf(stderr, "[DEBUG] Block %zu: Loaded overlap - first sample=%.6f, last sample=%.6f\n",
                            blockCount, overlapBuffer[0], overlapBuffer[overlapSize_-1]);
                }
            }

            // Copy new input data from upsampled signal
            if (inputPos + currentBlockSize <= outputFrames) {
                Utils::checkCudaError(
                    cudaMemcpyAsync(d_paddedInput + overlapSize_, d_upsampledInput + inputPos,
                                   currentBlockSize * sizeof(float), cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy block to padded"
                );
            }

            // Perform FFT convolution on this block
            Utils::checkCufftError(
                cufftExecR2C(fftPlanForward_, d_paddedInput, d_inputFFT),
                "cufftExecR2C block"
            );

            // Complex multiplication with pre-computed filter FFT
            // Note: complexMultiplyKernel modifies data in-place, so we operate directly on d_inputFFT
            threadsPerBlock = 256;
            blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;

            complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                d_inputFFT, d_filterFFT_, fftComplexSize
            );

            Utils::checkCudaError(
                cudaGetLastError(),
                "complexMultiplyKernel launch"
            );

            // Inverse FFT (operate on d_inputFFT since we modified it in-place)
            Utils::checkCufftError(
                cufftExecC2R(fftPlanInverse_, d_inputFFT, d_convResult),
                "cufftExecC2R block"
            );

            // Scale by FFT size
            float scale = 1.0f / fftSize_;
            int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
            scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
                d_convResult, fftSize_, scale
            );

            Utils::checkCudaError(
                cudaGetLastError(),
                "scaleKernel launch"
            );

            // Extract valid output (discard first overlapSize_ samples)
            size_t validOutputSize = (outputFrames - outputPos < static_cast<size_t>(validOutputPerBlock)) ?
                                      (outputFrames - outputPos) : static_cast<size_t>(validOutputPerBlock);

            Utils::checkCudaError(
                cudaMemcpyAsync(outputData.data() + outputPos, d_convResult + overlapSize_,
                               validOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpy valid output to host"
            );

            // Save overlap for next block
            // For Overlap-Save with very large filter (M=1000000), we have:
            //   FFT size N = 1048576
            //   Overlap = M-1 = 999999
            //   Valid output per block = N - (M-1) = 48577
            //
            // Each iteration processes:
            //   Input block: [previous overlap (999999) | new samples (48577)]
            //   Output: discard first 999999, keep last 48577
            //
            // The overlap for NEXT iteration should be the last 999999 samples
            // of the CURRENT input block, which is:
            //   Current input block spans: inputPos to inputPos+validOutputPerBlock
            //   But we need 999999 samples for overlap, and we only have 48577 new samples
            //   So overlap = [last (999999-48577) of previous overlap] + [all 48577 new samples]
            //   In the upsampled signal, this is: from (inputPos+48577-999999) to (inputPos+48577-1)
            //                                    = from (inputPos-951422) to (inputPos+48576)
            // Wait, that's going backwards!
            //
            // Let me reconsider: after processing block at inputPos, next block starts
            // at inputPos+validOutputPerBlock. The overlap needed is the 999999 samples
            // BEFORE that position, i.e., from (inputPos+validOutputPerBlock-overlap) to
            // (inputPos+validOutputPerBlock-1) = from (inputPos+48577-999999) to (inputPos+48576)
            //
            // But actually, we're already AT inputPos+validOutputPerBlock after the advance!
            // No wait, the advance happens AFTER this. So we save from inputPos+validOutputSize-overlapSize_+1
            //
            // Simpler: just save from (inputPos + validOutputSize) for overlapSize_ samples,
            // which is the END of current block extending into the next region.
            if (outputPos + validOutputSize < outputFrames) {
                // Next overlap: the overlapSize_ samples starting from where next block will begin
                size_t nextBlockStart = inputPos + validOutputSize;
                if (nextBlockStart >= overlapSize_ && nextBlockStart < outputFrames) {
                    // Get the last overlapSize_ samples before nextBlockStart
                    size_t overlapStart = nextBlockStart - overlapSize_;
                    if (overlapStart + overlapSize_ <= outputFrames) {
                        Utils::checkCudaError(
                            cudaMemcpyAsync(overlapBuffer.data(), d_upsampledInput + overlapStart,
                                           overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
                            "cudaMemcpy overlap from device"
                        );

                        // DEBUG: Log saved overlap positions (first few blocks only)
                        if (blockCount < 3) {
                            fprintf(stderr, "[DEBUG] Block %zu: Saved overlap from position %zu (nextBlockStart=%zu - overlapSize=%d)\n",
                                    blockCount, overlapStart, nextBlockStart, overlapSize_);
                        }
                    }
                }
            }

            // DEBUG: Log block processing summary (first few blocks only)
            if (blockCount < 3) {
                fprintf(stderr, "[DEBUG] Block %zu: inputPos=%zu, outputPos=%zu, validOutputSize=%zu, currentBlockSize=%zu\n",
                        blockCount, inputPos, outputPos, validOutputSize, currentBlockSize);
            }

            // Advance positions
            outputPos += validOutputSize;
            inputPos += validOutputSize;
            blockCount++;  // DEBUG: Increment block counter
        }

        // Synchronize stream before cleanup
        Utils::checkCudaError(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize"
        );

        // Cleanup temporary buffers
        cudaFree(d_upsampledInput);
        cudaFree(d_paddedInput);
        cudaFree(d_inputFFT);
        cudaFree(d_convResult);

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in processChannelWithStream: " << e.what() << std::endl;

        // Clean up all allocated GPU memory
        if (d_upsampledInput) cudaFree(d_upsampledInput);
        if (d_input) cudaFree(d_input);
        if (d_paddedInput) cudaFree(d_paddedInput);
        if (d_inputFFT) cudaFree(d_inputFFT);
        if (d_convResult) cudaFree(d_convResult);

        return false;
    }
}

bool GPUUpsampler::processStereo(const float* leftInput,
                                 const float* rightInput,
                                 size_t inputFrames,
                                 std::vector<float>& leftOutput,
                                 std::vector<float>& rightOutput) {
    auto startTime = std::chrono::high_resolution_clock::now();

    // Launch both channels in parallel using separate streams
    // Note: We need to use lambda or thread to truly parallelize,
    // but CUDA streams already allow concurrent execution on GPU

    // Process left channel on streamLeft_
    bool leftSuccess = processChannelWithStream(leftInput, inputFrames, leftOutput,
                                                streamLeft_, overlapBuffer_);

    // Process right channel on streamRight_ (can execute in parallel with left)
    bool rightSuccess = processChannelWithStream(rightInput, inputFrames, rightOutput,
                                                 streamRight_, overlapBufferRight_);

    // CRITICAL FIX: Explicitly synchronize both streams to prevent race conditions
    // Even though processChannelWithStream() synchronizes internally, we need to ensure
    // both channels complete before accessing shared resources or returning results
    Utils::checkCudaError(
        cudaStreamSynchronize(streamLeft_),
        "cudaStreamSynchronize left channel final"
    );
    Utils::checkCudaError(
        cudaStreamSynchronize(streamRight_),
        "cudaStreamSynchronize right channel final"
    );

    if (leftSuccess && rightSuccess) {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        stats_.totalProcessingTime += elapsed.count();
        stats_.framesProcessed += inputFrames * 2; // Count both channels
        stats_.gpuUtilization = Utils::getGPUUtilization();
    }

    return leftSuccess && rightSuccess;
}

void GPUUpsampler::cleanup() {
    if (d_filterCoeffs_) cudaFree(d_filterCoeffs_);
    if (d_filterFFT_) cudaFree(d_filterFFT_);
    if (d_inputBlock_) cudaFree(d_inputBlock_);
    if (d_outputBlock_) cudaFree(d_outputBlock_);
    if (d_inputFFT_) cudaFree(d_inputFFT_);
    if (d_convResult_) cudaFree(d_convResult_);

    // Free streaming buffers
    if (d_streamInput_) cudaFree(d_streamInput_);
    if (d_streamUpsampled_) cudaFree(d_streamUpsampled_);
    if (d_streamPadded_) cudaFree(d_streamPadded_);
    if (d_streamInputFFT_) cudaFree(d_streamInputFFT_);
    if (d_streamConvResult_) cudaFree(d_streamConvResult_);

    if (fftPlanForward_) cufftDestroy(fftPlanForward_);
    if (fftPlanInverse_) cufftDestroy(fftPlanInverse_);

    if (stream_) cudaStreamDestroy(stream_);
    if (streamLeft_) cudaStreamDestroy(streamLeft_);
    if (streamRight_) cudaStreamDestroy(streamRight_);

    d_filterCoeffs_ = nullptr;
    d_filterFFT_ = nullptr;
    d_inputBlock_ = nullptr;
    d_outputBlock_ = nullptr;
    d_inputFFT_ = nullptr;
    d_convResult_ = nullptr;
    d_streamInput_ = nullptr;
    d_streamUpsampled_ = nullptr;
    d_streamPadded_ = nullptr;
    d_streamInputFFT_ = nullptr;
    d_streamConvResult_ = nullptr;
    fftPlanForward_ = 0;
    fftPlanInverse_ = 0;
    stream_ = nullptr;
    streamLeft_ = nullptr;
    streamRight_ = nullptr;
}

// Streaming mode methods
bool GPUUpsampler::initializeStreaming() {
    if (fftSize_ == 0 || overlapSize_ == 0) {
        std::cerr << "ERROR: GPU resources not initialized. Call initialize() first." << std::endl;
        return false;
    }

    validOutputPerBlock_ = fftSize_ - overlapSize_;
    streamValidInputPerBlock_ = validOutputPerBlock_ / upsampleRatio_;

    // Pre-allocate GPU buffers to avoid malloc/free in real-time callbacks
    size_t upsampledSize = streamValidInputPerBlock_ * upsampleRatio_;
    int fftComplexSize = fftSize_ / 2 + 1;

    Utils::checkCudaError(
        cudaMalloc(&d_streamInput_, streamValidInputPerBlock_ * sizeof(float)),
        "cudaMalloc streaming input buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamUpsampled_, upsampledSize * sizeof(float)),
        "cudaMalloc streaming upsampled buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamPadded_, fftSize_ * sizeof(float)),
        "cudaMalloc streaming padded buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamInputFFT_, fftComplexSize * sizeof(cufftComplex)),
        "cudaMalloc streaming FFT buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamConvResult_, fftSize_ * sizeof(float)),
        "cudaMalloc streaming conv result buffer"
    );

    streamInitialized_ = true;

    fprintf(stderr, "[Streaming] Initialized:\n");
    fprintf(stderr, "  - Input samples needed per block: %zu (at input rate %d Hz)\n",
            streamValidInputPerBlock_, 44100);
    fprintf(stderr, "  - Output samples generated per block: %d (at output rate %d Hz)\n",
            validOutputPerBlock_, 705600);
    fprintf(stderr, "  - Latency: ~%.1f ms\n",
            1000.0 * streamValidInputPerBlock_ / 44100.0);
    fprintf(stderr, "  - GPU streaming buffers pre-allocated\n");

    return true;
}

void GPUUpsampler::resetStreaming() {
    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);
    std::fill(overlapBufferRight_.begin(), overlapBufferRight_.end(), 0.0f);
    fprintf(stderr, "[Streaming] Reset: overlap buffers cleared\n");
}

bool GPUUpsampler::processStreamBlock(const float* inputData,
                                       size_t inputFrames,
                                       std::vector<float>& outputData,
                                       cudaStream_t stream,
                                       std::vector<float>& streamInputBuffer,
                                       size_t& streamInputAccumulated) {
    if (!streamInitialized_) {
        std::cerr << "ERROR: Streaming mode not initialized. Call initializeStreaming() first." << std::endl;
        return false;
    }

    // 1. Accumulate input samples
    if (streamInputAccumulated + inputFrames > streamInputBuffer.size()) {
        std::cerr << "ERROR: Stream input buffer overflow" << std::endl;
        return false;
    }

    std::copy(inputData, inputData + inputFrames,
              streamInputBuffer.begin() + streamInputAccumulated);
    streamInputAccumulated += inputFrames;

    // 2. Check if we have enough samples for one block
    if (streamInputAccumulated < streamValidInputPerBlock_) {
        // Not enough data yet - return false (no output generated)
        outputData.clear();
        return false;
    }

    // 3. Process one block using pre-allocated GPU buffers
    size_t samplesToProcess = streamValidInputPerBlock_;
    size_t outputFrames = samplesToProcess * upsampleRatio_;

    // Step 3a: Transfer input to GPU using pre-allocated d_streamInput_
    Utils::checkCudaError(
        cudaMemcpyAsync(d_streamInput_, streamInputBuffer.data(), samplesToProcess * sizeof(float),
                       cudaMemcpyHostToDevice, stream),
        "cudaMemcpy streaming input to device"
    );

    // Step 3b: Zero-padding (upsampling) using pre-allocated d_streamUpsampled_
    int threadsPerBlock = 256;
    int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
    zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_streamInput_, d_streamUpsampled_, samplesToProcess, upsampleRatio_
    );

    // Step 3c: Overlap-Save FFT convolution using pre-allocated buffers
    int fftComplexSize = fftSize_ / 2 + 1;

    // Prepare input: [overlap | new samples] using pre-allocated d_streamPadded_
    Utils::checkCudaError(
        cudaMemsetAsync(d_streamPadded_, 0, fftSize_ * sizeof(float), stream),
        "cudaMemset streaming padded"
    );

    // Determine which overlap buffer to use based on which stream this is
    std::vector<float>& overlap = (stream == streamLeft_) ? overlapBuffer_ :
                                  (stream == streamRight_) ? overlapBufferRight_ : overlapBuffer_;

    if (overlapSize_ > 0) {
        Utils::checkCudaError(
            cudaMemcpyAsync(d_streamPadded_, overlap.data(),
                           overlapSize_ * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpy streaming overlap to device"
        );
    }

    Utils::checkCudaError(
        cudaMemcpyAsync(d_streamPadded_ + overlapSize_, d_streamUpsampled_,
                       outputFrames * sizeof(float), cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy streaming block to padded"
    );

    // FFT convolution using pre-allocated buffers
    Utils::checkCufftError(
        cufftSetStream(fftPlanForward_, stream),
        "cufftSetStream forward"
    );

    Utils::checkCufftError(
        cufftExecR2C(fftPlanForward_, d_streamPadded_, d_streamInputFFT_),
        "cufftExecR2C streaming"
    );

    threadsPerBlock = 256;
    blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
    complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_streamInputFFT_, d_filterFFT_, fftComplexSize
    );

    Utils::checkCufftError(
        cufftSetStream(fftPlanInverse_, stream),
        "cufftSetStream inverse"
    );

    Utils::checkCufftError(
        cufftExecC2R(fftPlanInverse_, d_streamInputFFT_, d_streamConvResult_),
        "cufftExecC2R streaming"
    );

    // Scale
    float scale = 1.0f / fftSize_;
    int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
        d_streamConvResult_, fftSize_, scale
    );

    // Extract valid output (discard first overlapSize_ samples)
    outputData.resize(validOutputPerBlock_);
    Utils::checkCudaError(
        cudaMemcpyAsync(outputData.data(), d_streamConvResult_ + overlapSize_,
                       validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpy streaming output to host"
    );

    // Save overlap for next block
    // CRITICAL: Must save from d_streamPadded_ (overlap+new concatenated), not d_streamUpsampled_ (new only)
    // Save the LAST overlapSize_ samples from the concatenated buffer [overlap | new]
    // Total size = overlapSize_ + outputFrames, so start at outputFrames
    if (fftSize_ >= overlapSize_) {
        Utils::checkCudaError(
            cudaMemcpyAsync(overlap.data(), d_streamPadded_ + (fftSize_ - overlapSize_),
                           overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpy streaming overlap from device"
        );
    }

    // Synchronize stream to ensure all operations complete
    Utils::checkCudaError(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize streaming"
    );

    // 4. Shift remaining samples in input buffer
    size_t remaining = streamInputAccumulated - samplesToProcess;
    if (remaining > 0) {
        std::copy(streamInputBuffer.begin() + samplesToProcess,
                  streamInputBuffer.begin() + streamInputAccumulated,
                  streamInputBuffer.begin());
    }
    streamInputAccumulated = remaining;

    return true; // Output was generated
}

} // namespace ConvolutionEngine
