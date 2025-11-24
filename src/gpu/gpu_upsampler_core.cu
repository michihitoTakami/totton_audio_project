#include "convolution_engine.h"
#include "gpu/convolution_kernels.h"
#include "gpu/cuda_utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>

namespace ConvolutionEngine {

// GPUUpsampler implementation - Core methods

void GPUUpsampler::resizeOverlapBuffers(size_t newSize) {
    unregisterHostBuffers();
    overlapBuffer_.assign(newSize, 0.0f);
    overlapBufferRight_.assign(newSize, 0.0f);

    registerHostBuffer(overlapBuffer_.data(),
                       overlapBuffer_.size() * sizeof(float),
                       "cudaHostRegister overlap buffer (L)");
    registerHostBuffer(overlapBufferRight_.data(),
                       overlapBufferRight_.size() * sizeof(float),
                       "cudaHostRegister overlap buffer (R)");
}

GPUUpsampler::GPUUpsampler()
    : upsampleRatio_(1), blockSize_(8192), filterTaps_(0), fftSize_(0),
      d_filterCoeffs_(nullptr),
      dualRateEnabled_(false), currentRateFamily_(RateFamily::RATE_44K),
      d_filterFFT_44k_(nullptr), d_filterFFT_48k_(nullptr),
      quadPhaseEnabled_(false),
      d_filterFFT_44k_linear_(nullptr), d_filterFFT_48k_linear_(nullptr),
      multiRateEnabled_(false), currentInputRate_(44100), currentMultiRateIndex_(0),
      d_filterFFT_A_(nullptr), d_filterFFT_B_(nullptr), d_activeFilterFFT_(nullptr),
      d_originalFilterFFT_(nullptr), filterFftSize_(0), eqApplied_(false),
      d_inputBlock_(nullptr), d_outputBlock_(nullptr),
      d_inputFFT_(nullptr), d_convResult_(nullptr),
      fftPlanForward_(0), fftPlanInverse_(0),
      eqPlanD2Z_(0), eqPlanZ2D_(0), d_eqLogMag_(nullptr), d_eqComplexSpec_(nullptr),
      overlapSize_(0), stream_(nullptr), streamLeft_(nullptr), streamRight_(nullptr),
      streamValidInputPerBlock_(0), streamInitialized_(false), validOutputPerBlock_(0),
      streamOverlapSize_(0),
      d_streamInput_(nullptr), d_streamUpsampled_(nullptr), d_streamPadded_(nullptr),
      d_streamInputFFT_(nullptr), d_streamConvResult_(nullptr),
      d_overlapLeft_(nullptr), d_overlapRight_(nullptr),
      pinnedStreamInputLeft_(nullptr), pinnedStreamInputRight_(nullptr),
      pinnedStreamInputMono_(nullptr),
      pinnedStreamInputLeftBytes_(0), pinnedStreamInputRightBytes_(0),
      pinnedStreamInputMonoBytes_(0),
      pinnedStreamOutputLeft_(nullptr), pinnedStreamOutputRight_(nullptr),
      pinnedStreamOutputMono_(nullptr),
      pinnedStreamOutputLeftBytes_(0), pinnedStreamOutputRightBytes_(0),
      pinnedStreamOutputMonoBytes_(0) {
    stats_ = Stats();
    // Initialize multi-rate FFT pointers to nullptr
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        d_filterFFT_Multi_[i] = nullptr;
    }
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
        resizeOverlapBuffers(overlapSize_);

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

        filterFftSize_ = fftComplexSize;

        // Allocate double-buffered filter FFT (ping-pong) for glitch-free EQ updates
        Utils::checkCudaError(
            cudaMalloc(&d_filterFFT_A_, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc filter FFT A"
        );
        Utils::checkCudaError(
            cudaMalloc(&d_filterFFT_B_, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc filter FFT B"
        );
        d_activeFilterFFT_ = d_filterFFT_A_;  // Start with buffer A

        // Allocate backup for original filter FFT (for EQ restore)
        Utils::checkCudaError(
            cudaMalloc(&d_originalFilterFFT_, fftComplexSize * sizeof(cufftComplex)),
            "cudaMalloc original filter FFT"
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

        // Compute filter FFT into buffer A
        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_filterPadded, d_filterFFT_A_),
            "cufftExecR2C filter"
        );

        // Copy to buffer B (both start with same initial filter)
        Utils::checkCudaError(
            cudaMemcpy(d_filterFFT_B_, d_filterFFT_A_,
                      filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy filter FFT A to B"
        );

        // Backup original filter FFT for EQ restore
        Utils::checkCudaError(
            cudaMemcpy(d_originalFilterFFT_, d_filterFFT_A_,
                      filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy backup original filter FFT"
        );

        // Host cache for original filter FFT (avoids D→H copy during EQ application)
        h_originalFilterFft_.resize(filterFftSize_);
        Utils::checkCudaError(
            cudaMemcpy(h_originalFilterFft_.data(), d_filterFFT_A_,
                      filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
            "cudaMemcpy original filter to host cache"
        );

        // Pre-allocate EQ-specific resources (persistent for real-time EQ switching)
        Utils::checkCufftError(
            cufftPlan1d(&eqPlanD2Z_, fftSize_, CUFFT_D2Z, 1),
            "cufftPlan1d EQ D2Z"
        );
        Utils::checkCufftError(
            cufftPlan1d(&eqPlanZ2D_, fftSize_, CUFFT_Z2D, 1),
            "cufftPlan1d EQ Z2D"
        );
        Utils::checkCudaError(
            cudaMalloc(&d_eqLogMag_, fftSize_ * sizeof(cufftDoubleReal)),
            "cudaMalloc EQ log magnitude buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&d_eqComplexSpec_, filterFftSize_ * sizeof(cufftDoubleComplex)),
            "cudaMalloc EQ complex spectrum buffer"
        );

        cudaFree(d_filterPadded);
        eqApplied_ = false;

        std::cout << "  GPU resources allocated successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in setupGPUResources: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void GPUUpsampler::registerHostBuffer(void* ptr, size_t bytes, const char* context) {
    if (!ptr || bytes == 0) {
        return;
    }

    for (const auto& buf : pinnedHostBuffers_) {
        if (buf.ptr == ptr) {
            return;  // Already registered
        }
    }

    Utils::checkCudaError(
        cudaHostRegister(ptr, bytes, cudaHostRegisterDefault),
        context
    );
    pinnedHostBuffers_.push_back({ptr, bytes});
}

void GPUUpsampler::removePinnedHostBuffer(void* ptr) {
    pinnedHostBuffers_.erase(
        std::remove_if(pinnedHostBuffers_.begin(), pinnedHostBuffers_.end(),
                       [ptr](const PinnedHostBuffer& buf) { return buf.ptr == ptr; }),
        pinnedHostBuffers_.end()
    );
}

void GPUUpsampler::registerStreamInputBuffer(std::vector<float>& buffer, cudaStream_t stream) {
    if (buffer.empty()) {
        return;
    }

    void* ptr = buffer.data();
    size_t bytes = buffer.size() * sizeof(float);

    // Track per-stream host buffers to avoid duplicate registrations when vectors reallocate
    void** trackedPtr = nullptr;
    size_t* trackedBytes = nullptr;
    const char* context = nullptr;

    if (stream == streamLeft_) {
        trackedPtr = &pinnedStreamInputLeft_;
        trackedBytes = &pinnedStreamInputLeftBytes_;
        context = "cudaHostRegister streaming input buffer (L)";
    } else if (stream == streamRight_) {
        trackedPtr = &pinnedStreamInputRight_;
        trackedBytes = &pinnedStreamInputRightBytes_;
        context = "cudaHostRegister streaming input buffer (R)";
    } else {
        trackedPtr = &pinnedStreamInputMono_;
        trackedBytes = &pinnedStreamInputMonoBytes_;
        context = "cudaHostRegister streaming input buffer (mono)";
    }

    if (*trackedPtr == ptr && *trackedBytes == bytes) {
        return;  // Already registered for this stream
    }

    if (*trackedPtr) {
        cudaHostUnregister(*trackedPtr);
        removePinnedHostBuffer(*trackedPtr);
    }

    registerHostBuffer(ptr, bytes, context);
    *trackedPtr = ptr;
    *trackedBytes = bytes;
}

void GPUUpsampler::registerStreamOutputBuffer(std::vector<float>& buffer, cudaStream_t stream) {
    if (buffer.empty()) {
        return;
    }

    void* ptr = buffer.data();
    size_t bytes = buffer.size() * sizeof(float);

    void** trackedPtr = nullptr;
    size_t* trackedBytes = nullptr;
    const char* context = nullptr;

    if (stream == streamLeft_) {
        trackedPtr = &pinnedStreamOutputLeft_;
        trackedBytes = &pinnedStreamOutputLeftBytes_;
        context = "cudaHostRegister streaming output buffer (L)";
    } else if (stream == streamRight_) {
        trackedPtr = &pinnedStreamOutputRight_;
        trackedBytes = &pinnedStreamOutputRightBytes_;
        context = "cudaHostRegister streaming output buffer (R)";
    } else {
        trackedPtr = &pinnedStreamOutputMono_;
        trackedBytes = &pinnedStreamOutputMonoBytes_;
        context = "cudaHostRegister streaming output buffer (mono)";
    }

    if (*trackedPtr == ptr && *trackedBytes == bytes) {
        return;  // Already registered
    }

    if (*trackedPtr) {
        cudaHostUnregister(*trackedPtr);
        removePinnedHostBuffer(*trackedPtr);
    }

    registerHostBuffer(ptr, bytes, context);
    *trackedPtr = ptr;
    *trackedBytes = bytes;
}

void GPUUpsampler::unregisterHostBuffers() {
    for (const auto& buf : pinnedHostBuffers_) {
        cudaHostUnregister(buf.ptr);
    }
    pinnedHostBuffers_.clear();
    pinnedStreamInputLeft_ = nullptr;
    pinnedStreamInputRight_ = nullptr;
    pinnedStreamInputMono_ = nullptr;
    pinnedStreamInputLeftBytes_ = 0;
    pinnedStreamInputRightBytes_ = 0;
    pinnedStreamInputMonoBytes_ = 0;
    pinnedStreamOutputLeft_ = nullptr;
    pinnedStreamOutputRight_ = nullptr;
    pinnedStreamOutputMono_ = nullptr;
    pinnedStreamOutputLeftBytes_ = 0;
    pinnedStreamOutputRightBytes_ = 0;
    pinnedStreamOutputMonoBytes_ = 0;
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
        ScopedHostPin outputPinned(outputData.data(),
                                   outputFrames * sizeof(float),
                                   "cudaHostRegister output buffer (offline)");

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
        int validOutputPerBlock = fftSize_ - overlapSize_;

        // Process audio in blocks
        size_t outputPos = 0;
        size_t inputPos = 0;
        size_t blockCount = 0;

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
                cufftSetStream(fftPlanForward_, stream),
                "cufftSetStream forward (offline)"
            );
            Utils::checkCufftError(
                cufftExecR2C(fftPlanForward_, d_paddedInput, d_inputFFT),
                "cufftExecR2C block"
            );

            // Complex multiplication with pre-computed filter FFT
            threadsPerBlock = 256;
            blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;

            complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                d_inputFFT, d_activeFilterFFT_, fftComplexSize
            );

            Utils::checkCudaError(
                cudaGetLastError(),
                "complexMultiplyKernel launch"
            );

            // Inverse FFT
            Utils::checkCufftError(
                cufftSetStream(fftPlanInverse_, stream),
                "cufftSetStream inverse (offline)"
            );
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
            Utils::checkCudaError(
                cudaMemcpyAsync(overlapBuffer.data(), d_paddedInput + validOutputPerBlock,
                               overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpy overlap from device"
            );

            if (blockCount < 3) {
                fprintf(stderr, "[DEBUG] Block %zu: inputPos=%zu, outputPos=%zu, validOutputSize=%zu, currentBlockSize=%zu\n",
                        blockCount, inputPos, outputPos, validOutputSize, currentBlockSize);
            }

            // Advance positions
            outputPos += validOutputSize;
            inputPos += validOutputSize;
            blockCount++;
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

    // Process left channel on streamLeft_
    bool leftSuccess = processChannelWithStream(leftInput, inputFrames, leftOutput,
                                                streamLeft_, overlapBuffer_);

    // Process right channel on streamRight_ (can execute in parallel with left)
    bool rightSuccess = processChannelWithStream(rightInput, inputFrames, rightOutput,
                                                 streamRight_, overlapBufferRight_);

    // Synchronize both streams
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
        stats_.framesProcessed += inputFrames * 2;
        stats_.gpuUtilization = Utils::getGPUUtilization();
    }

    return leftSuccess && rightSuccess;
}

void GPUUpsampler::cleanup() {
    unregisterHostBuffers();

    if (d_filterCoeffs_) cudaFree(d_filterCoeffs_);
    // Free dual-rate filter FFT buffers
    if (d_filterFFT_44k_) cudaFree(d_filterFFT_44k_);
    if (d_filterFFT_48k_) cudaFree(d_filterFFT_48k_);
    // Free quad-phase (linear phase) filter FFT buffers
    if (d_filterFFT_44k_linear_) cudaFree(d_filterFFT_44k_linear_);
    if (d_filterFFT_48k_linear_) cudaFree(d_filterFFT_48k_linear_);
    // Free multi-rate filter FFT buffers
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        if (d_filterFFT_Multi_[i]) cudaFree(d_filterFFT_Multi_[i]);
    }
    // Free double-buffered filter FFT (ping-pong)
    if (d_filterFFT_A_) cudaFree(d_filterFFT_A_);
    if (d_filterFFT_B_) cudaFree(d_filterFFT_B_);
    if (d_originalFilterFFT_) cudaFree(d_originalFilterFFT_);
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
    if (d_overlapLeft_) cudaFree(d_overlapLeft_);
    if (d_overlapRight_) cudaFree(d_overlapRight_);

    if (fftPlanForward_) cufftDestroy(fftPlanForward_);
    if (fftPlanInverse_) cufftDestroy(fftPlanInverse_);

    // Free EQ-specific resources
    if (eqPlanD2Z_) cufftDestroy(eqPlanD2Z_);
    if (eqPlanZ2D_) cufftDestroy(eqPlanZ2D_);
    if (d_eqLogMag_) cudaFree(d_eqLogMag_);
    if (d_eqComplexSpec_) cudaFree(d_eqComplexSpec_);
    h_originalFilterFft_.clear();

    if (stream_) cudaStreamDestroy(stream_);
    if (streamLeft_) cudaStreamDestroy(streamLeft_);
    if (streamRight_) cudaStreamDestroy(streamRight_);

    d_filterCoeffs_ = nullptr;
    // Dual-rate cleanup
    d_filterFFT_44k_ = nullptr;
    d_filterFFT_48k_ = nullptr;
    dualRateEnabled_ = false;
    currentRateFamily_ = RateFamily::RATE_44K;
    h_filterCoeffs44k_.clear();
    h_filterCoeffs48k_.clear();
    // Quad-phase cleanup
    quadPhaseEnabled_ = false;
    d_filterFFT_44k_linear_ = nullptr;
    d_filterFFT_48k_linear_ = nullptr;
    h_filterCoeffs44k_linear_.clear();
    h_filterCoeffs48k_linear_.clear();
    // Multi-rate cleanup
    multiRateEnabled_ = false;
    currentInputRate_ = 44100;
    currentMultiRateIndex_ = 0;
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        d_filterFFT_Multi_[i] = nullptr;
        h_filterCoeffsMulti_[i].clear();
    }
    // Double-buffer cleanup
    d_filterFFT_A_ = nullptr;
    d_filterFFT_B_ = nullptr;
    d_activeFilterFFT_ = nullptr;
    d_originalFilterFFT_ = nullptr;
    filterFftSize_ = 0;
    eqApplied_ = false;
    d_inputBlock_ = nullptr;
    d_outputBlock_ = nullptr;
    d_inputFFT_ = nullptr;
    d_convResult_ = nullptr;
    d_streamInput_ = nullptr;
    d_streamUpsampled_ = nullptr;
    d_streamPadded_ = nullptr;
    d_streamInputFFT_ = nullptr;
    d_streamConvResult_ = nullptr;
    d_overlapLeft_ = nullptr;
    d_overlapRight_ = nullptr;
    fftPlanForward_ = 0;
    fftPlanInverse_ = 0;
    stream_ = nullptr;
    streamLeft_ = nullptr;
    streamRight_ = nullptr;
    streamValidInputPerBlock_ = 0;
    validOutputPerBlock_ = 0;
    streamOverlapSize_ = 0;
    streamInitialized_ = false;
}

}  // namespace ConvolutionEngine
