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

using Precision = ActivePrecisionTraits;
using Sample = DeviceSample;
using Complex = DeviceFftComplex;
using ScaleType = DeviceScale;

inline cudaError_t copyHostToDeviceSamplesAsync(Sample* dst, const float* src, size_t count,
                                                cudaStream_t stream) {
    if constexpr (Precision::kIsDouble) {
        std::vector<Sample> temp(count);
        for (size_t i = 0; i < count; ++i) {
            temp[i] = static_cast<Sample>(src[i]);
        }
        return cudaMemcpyAsync(dst, temp.data(), count * sizeof(Sample),
                               cudaMemcpyHostToDevice, stream);
    }
    return cudaMemcpyAsync(dst, src, count * sizeof(Sample),
                           cudaMemcpyHostToDevice, stream);
}

inline cudaError_t copyDeviceToHostSamplesSync(float* dst, const Sample* src, size_t count) {
    return copyDeviceToHostSamples<Precision>(dst, src, count);
}

inline cudaError_t copyDeviceToHostSamplesAsync(float* dst, const Sample* src, size_t count,
                                                cudaStream_t stream) {
    if constexpr (Precision::kIsDouble) {
        std::vector<Sample> temp(count);
        auto err = cudaMemcpyAsync(temp.data(), src, count * sizeof(Sample),
                                   cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return err;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return err;
        }
        for (size_t i = 0; i < count; ++i) {
            dst[i] = static_cast<float>(temp[i]);
        }
        return cudaSuccess;
    }
    return cudaMemcpyAsync(dst, src, count * sizeof(Sample),
                           cudaMemcpyDeviceToHost, stream);
}

float* GPUUpsampler::getOutputScratch(cudaStream_t stream) {
    if constexpr (!Precision::kIsDouble) {
        return nullptr;
    }
    if (stream == streamLeft_) {
        return d_outputScratchLeft_;
    }
    if (stream == streamRight_) {
        return d_outputScratchRight_;
    }
    return d_outputScratch_;
}

cudaError_t GPUUpsampler::downconvertToHost(float* hostDst, const Sample* deviceSrc, size_t count,
                                            cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }

    if constexpr (Precision::kIsDouble) {
        float* scratch = getOutputScratch(stream);
        if (!scratch) {
            return cudaErrorInvalidDevicePointer;
        }

        int threadsPerBlock = 256;
        int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
        downconvertToFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            deviceSrc, scratch, static_cast<int>(count), -1.0f, 1.0f);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        return cudaMemcpyAsync(hostDst, scratch, count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
    }

    return cudaMemcpyAsync(hostDst, deviceSrc, count * sizeof(float),
                           cudaMemcpyDeviceToHost, stream);
}

cudaError_t GPUUpsampler::downconvertToHostSync(float* hostDst, const Sample* deviceSrc,
                                                size_t count) {
    if (count == 0) {
        return cudaSuccess;
    }

    if constexpr (Precision::kIsDouble) {
        float* scratch = d_outputScratch_;
        if (!scratch) {
            return cudaErrorInvalidDevicePointer;
        }

        int threadsPerBlock = 256;
        int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
        downconvertToFloatKernel<<<blocks, threadsPerBlock>>>(deviceSrc, scratch,
                                                              static_cast<int>(count),
                                                              -1.0f, 1.0f);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        return cudaMemcpy(hostDst, scratch, count * sizeof(float), cudaMemcpyDeviceToHost);
    }

    return cudaMemcpy(hostDst, deviceSrc, count * sizeof(float), cudaMemcpyDeviceToHost);
}

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
      d_originalFilterFFT_(nullptr), d_crossfadeFilterSnapshot_(nullptr),
      filterFftSize_(0), eqApplied_(false),
      d_inputBlock_(nullptr), d_outputBlock_(nullptr),
      d_inputFFT_(nullptr), d_convResult_(nullptr),
      fftPlanForward_(0), fftPlanInverse_(0),
      eqPlanD2Z_(0), eqPlanZ2D_(0), d_eqLogMag_(nullptr), d_eqComplexSpec_(nullptr),
      overlapSize_(0), stream_(nullptr), streamLeft_(nullptr), streamRight_(nullptr),
      streamValidInputPerBlock_(0), streamInitialized_(false), validOutputPerBlock_(0),
      streamOverlapSize_(0),
      d_streamInput_(nullptr), d_streamUpsampled_(nullptr), d_streamPadded_(nullptr),
      d_streamInputFFT_(nullptr), d_streamConvResult_(nullptr),
      d_streamInputFFTBackup_(nullptr), d_streamConvResultOld_(nullptr),
      d_outputScratch_(nullptr), d_outputScratchLeft_(nullptr), d_outputScratchRight_(nullptr),
      d_overlapLeft_(nullptr), d_overlapRight_(nullptr),
      pinnedStreamInputLeft_(nullptr), pinnedStreamInputRight_(nullptr),
      pinnedStreamInputMono_(nullptr),
      pinnedStreamInputLeftBytes_(0), pinnedStreamInputRightBytes_(0),
      pinnedStreamInputMonoBytes_(0),
      pinnedStreamOutputLeft_(nullptr), pinnedStreamOutputRight_(nullptr),
      pinnedStreamOutputMono_(nullptr),
      pinnedStreamOutputLeftBytes_(0), pinnedStreamOutputRightBytes_(0),
      pinnedStreamOutputMonoBytes_(0),
      partitionFastIndex_(0), maxPartitionValidOutput_(0), partitionFastFftSize_(0),
      partitionFastFftComplexSize_(0), d_tailAccumulator_(nullptr), d_tailMixBuffer_(nullptr),
      d_upsampledHistory_(nullptr), tailAccumulatorSize_(0), historyBufferSize_(0),
      tailBaseSample_(0), tailBaseIndex_(0), partitionProcessedSamples_(0),
      partitionOutputSamples_(0), historyWriteIndex_(0), partitionStreamingInitialized_(false) {
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

    // Release CPU-side coefficient memory after GPU transfer (Jetson optimization)
    // FFT spectra are now on GPU; time-domain coefficients are no longer needed
    releaseHostCoefficients();

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
    baseFilterCentroid_ = PhaseAlignment::computeEnergyCentroid(h_filterCoeffs_);
    h_filterCoeffsTyped_ = convertCoefficientsToPrecision<Precision>(h_filterCoeffs_);
    return true;
}

void GPUUpsampler::setPartitionedConvolutionConfig(
    const AppConfig::PartitionedConvolutionConfig& config) {
    partitionConfig_ = config;
    partitionPlan_ = PartitionPlan{};

    if (!partitionConfig_.enabled) {
        return;
    }

    if (filterTaps_ <= 0 || upsampleRatio_ <= 0) {
        std::cout << "[Partition] Config stored (waiting for filter initialization)" << std::endl;
        return;
    }

    partitionPlan_ = buildPartitionPlan(filterTaps_, upsampleRatio_, partitionConfig_);
    if (!partitionPlan_.enabled || partitionPlan_.partitions.empty()) {
        std::cerr << "[Partition] Failed to build partition plan (taps=" << filterTaps_ << ")"
                  << std::endl;
        partitionPlan_ = PartitionPlan{};
        return;
    }

    int outputRate = getOutputSampleRate();
    std::cout << "[Partition] Enabled: " << partitionPlan_.describe(outputRate) << std::endl;
}

bool GPUUpsampler::setupGPUResources() {
    try {
        std::cout << "Setting up GPU resources..." << std::endl;

        // Create CUDA streams (one for mono, two for stereo parallel processing)
        bool highPriorityPrimary = false;
        stream_ = Utils::createPrioritizedStream("cudaStreamCreate primary",
                                                 cudaStreamNonBlocking, &highPriorityPrimary);

        bool highPriorityLeft = false;
        streamLeft_ = Utils::createPrioritizedStream("cudaStreamCreate left",
                                                     cudaStreamNonBlocking, &highPriorityLeft);

        bool highPriorityRight = false;
        streamRight_ = Utils::createPrioritizedStream("cudaStreamCreate right",
                                                      cudaStreamNonBlocking, &highPriorityRight);

        if (highPriorityPrimary || highPriorityLeft || highPriorityRight) {
            std::cout << "  CUDA streams configured with high priority scheduling" << std::endl;
        } else {
            std::cout << "  CUDA streams using default priority" << std::endl;
        }

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
            cudaMalloc(&d_filterCoeffs_, filterTaps_ * sizeof(Sample)),
            "cudaMalloc filter coefficients"
        );

        Utils::checkCudaError(
            copyHostToDeviceSamples<Precision>(
                d_filterCoeffs_, h_filterCoeffs_.data(), static_cast<size_t>(filterTaps_)),
            "cudaMemcpy filter coefficients"
        );

        // Allocate working buffers
        size_t upsampledBlockSize = blockSize_ * upsampleRatio_;

        Utils::checkCudaError(
            cudaMalloc(&d_inputBlock_, blockSize_ * sizeof(Sample)),
            "cudaMalloc input block"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_outputBlock_, (upsampledBlockSize + filterTaps_) * sizeof(Sample)),
            "cudaMalloc output block"
        );

        // Allocate FFT buffers
        int fftComplexSize = fftSize_ / 2 + 1;

        Utils::checkCudaError(
            cudaMalloc(&d_inputFFT_, fftComplexSize * sizeof(Complex)),
            "cudaMalloc input FFT"
        );

        filterFftSize_ = fftComplexSize;

        // Allocate double-buffered filter FFT (ping-pong) for glitch-free EQ updates
        Utils::checkCudaError(
            cudaMalloc(&d_filterFFT_A_, fftComplexSize * sizeof(Complex)),
            "cudaMalloc filter FFT A"
        );
        Utils::checkCudaError(
            cudaMalloc(&d_filterFFT_B_, fftComplexSize * sizeof(Complex)),
            "cudaMalloc filter FFT B"
        );
        d_activeFilterFFT_ = d_filterFFT_A_;  // Start with buffer A

        // Allocate backup for original filter FFT (for EQ restore)
        Utils::checkCudaError(
            cudaMalloc(&d_originalFilterFFT_, fftComplexSize * sizeof(Complex)),
            "cudaMalloc original filter FFT"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_convResult_, fftComplexSize * sizeof(Complex)),
            "cudaMalloc convolution result"
        );

        if constexpr (Precision::kIsDouble) {
            size_t scratchBytes = static_cast<size_t>(fftSize_) * sizeof(float);
            Utils::checkCudaError(
                cudaMalloc(&d_outputScratch_, scratchBytes),
                "cudaMalloc output scratch (primary)"
            );
            Utils::checkCudaError(
                cudaMalloc(&d_outputScratchLeft_, scratchBytes),
                "cudaMalloc output scratch (left)"
            );
            Utils::checkCudaError(
                cudaMalloc(&d_outputScratchRight_, scratchBytes),
                "cudaMalloc output scratch (right)"
            );
        }

        // Create cuFFT plans
        Utils::checkCufftError(
            cufftPlan1d(&fftPlanForward_, fftSize_, Precision::kFftTypeForward, 1),
            "cufftPlan1d forward"
        );

        Utils::checkCufftError(
            cufftPlan1d(&fftPlanInverse_, fftSize_, Precision::kFftTypeInverse, 1),
            "cufftPlan1d inverse"
        );

        Utils::checkCufftError(
            cufftPlan1d(&partitionImpulsePlanInverse_, fftSize_, Precision::kFftTypeInverse, 1),
            "cufftPlan1d partition impulse inverse"
        );

        // Pre-compute filter FFT
        Sample* d_filterPadded;
        Utils::checkCudaError(
            cudaMalloc(&d_filterPadded, fftSize_ * sizeof(Sample)),
            "cudaMalloc filter padded"
        );

        Utils::checkCudaError(
            cudaMemset(d_filterPadded, 0, fftSize_ * sizeof(Sample)),
            "cudaMemset filter padded"
        );

        Utils::checkCudaError(
            cudaMemcpy(d_filterPadded, d_filterCoeffs_,
                      filterTaps_ * sizeof(Sample), cudaMemcpyDeviceToDevice),
            "cudaMemcpy filter to padded"
        );

        // Compute filter FFT into buffer A
        Utils::checkCufftError(
            Precision::execForward(fftPlanForward_, d_filterPadded, d_filterFFT_A_),
            "cufftExecR2C filter"
        );

        // Copy to buffer B (both start with same initial filter)
        Utils::checkCudaError(
            cudaMemcpy(d_filterFFT_B_, d_filterFFT_A_,
                      filterFftSize_ * sizeof(Complex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy filter FFT A to B"
        );

        // Backup original filter FFT for EQ restore
        Utils::checkCudaError(
            cudaMemcpy(d_originalFilterFFT_, d_filterFFT_A_,
                      filterFftSize_ * sizeof(Complex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy backup original filter FFT"
        );

        // Host cache for original filter FFT (avoids D→H copy during EQ application)
        h_originalFilterFft_.resize(filterFftSize_);
        Utils::checkCudaError(
            cudaMemcpy(h_originalFilterFft_.data(), d_filterFFT_A_,
                      filterFftSize_ * sizeof(Complex), cudaMemcpyDeviceToHost),
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

        if (partitionConfig_.enabled) {
            setPartitionedConvolutionConfig(partitionConfig_);
        }

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

void GPUUpsampler::registerStreamInputBuffer(StreamFloatVector& buffer, cudaStream_t stream) {
    if (buffer.empty()) {
        return;
    }

    void* ptr = buffer.data();
    size_t bytes = buffer.size() * sizeof(float);

    // Track per-stream host buffers to avoid duplicate registrations when vectors reallocate
    void** trackedPtr = nullptr;
    size_t* trackedBytes = nullptr;
    if (stream == streamLeft_) {
        trackedPtr = &pinnedStreamInputLeft_;
        trackedBytes = &pinnedStreamInputLeftBytes_;
    } else if (stream == streamRight_) {
        trackedPtr = &pinnedStreamInputRight_;
        trackedBytes = &pinnedStreamInputRightBytes_;
    } else {
        trackedPtr = &pinnedStreamInputMono_;
        trackedBytes = &pinnedStreamInputMonoBytes_;
    }

    if (*trackedPtr == ptr && *trackedBytes == bytes) {
        return;  // Already prepared for this stream
    }

    *trackedPtr = ptr;
    *trackedBytes = bytes;
}

void GPUUpsampler::registerStreamOutputBuffer(StreamFloatVector& buffer, cudaStream_t stream) {
    if (buffer.empty()) {
        return;
    }

    void* ptr = buffer.data();
    size_t bytes = buffer.size() * sizeof(float);

    void** trackedPtr = nullptr;
    size_t* trackedBytes = nullptr;
    if (stream == streamLeft_) {
        trackedPtr = &pinnedStreamOutputLeft_;
        trackedBytes = &pinnedStreamOutputLeftBytes_;
    } else if (stream == streamRight_) {
        trackedPtr = &pinnedStreamOutputRight_;
        trackedBytes = &pinnedStreamOutputRightBytes_;
    } else {
        trackedPtr = &pinnedStreamOutputMono_;
        trackedBytes = &pinnedStreamOutputMonoBytes_;
    }

    if (*trackedPtr == ptr && *trackedBytes == bytes) {
        return;  // Already prepared
    }

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
    // Bypass mode: ratio 1 means input is already at output rate
    // Skip GPU convolution ONLY if EQ is not applied
    // When EQ is applied, we need convolution to apply the EQ filter
    if (upsampleRatio_ == 1 && !eqApplied_) {
        outputData.assign(inputData, inputData + inputFrames);
        return true;
    }

    // Initialize all GPU pointers to nullptr for safe cleanup
    Sample* d_upsampledInput = nullptr;
    Sample* d_input = nullptr;
    Sample* d_paddedInput = nullptr;
    Complex* d_inputFFT = nullptr;
    Sample* d_convResult = nullptr;

    try {
        size_t outputFrames = inputFrames * upsampleRatio_;
        outputData.resize(outputFrames, 0.0f);
        ScopedHostPin outputPinned(outputData.data(),
                                   outputFrames * sizeof(float),
                                   "cudaHostRegister output buffer (offline)");

        // Step 1: Zero-pad input signal (upsample) in one go
        Utils::checkCudaError(
            cudaMalloc(&d_upsampledInput, outputFrames * sizeof(Sample)),
            "cudaMalloc upsampled input"
        );

        Utils::checkCudaError(
            cudaMemset(d_upsampledInput, 0, outputFrames * sizeof(Sample)),
            "cudaMemset upsampled input"
        );

        // Copy input to device
        Utils::checkCudaError(
            cudaMalloc(&d_input, inputFrames * sizeof(Sample)),
            "cudaMalloc input"
        );

        Utils::checkCudaError(
            copyHostToDeviceSamplesAsync(d_input, inputData, inputFrames, stream),
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
            cudaMalloc(&d_paddedInput, fftSize_ * sizeof(Sample)),
            "cudaMalloc padded input"
        );

        int fftComplexSize = fftSize_ / 2 + 1;
        Utils::checkCudaError(
            cudaMalloc(&d_inputFFT, fftComplexSize * sizeof(Complex)),
            "cudaMalloc input FFT"
        );

        Utils::checkCudaError(
            cudaMalloc(&d_convResult, fftSize_ * sizeof(Sample)),
            "cudaMalloc conv result"
        );

        // Overlap-Save parameters
        // For short, single-block processing (output shorter than overlap), discard
        // only the filter energy centroid so that leading output (e.g., stereo
        // impulse offsets) remains in view.
        bool shortInput = outputFrames <= static_cast<size_t>(overlapSize_);
        size_t centroidOverlap = static_cast<size_t>(
            std::min<long double>(static_cast<long double>(overlapSize_),
                                  static_cast<long double>(std::max(0.0f, baseFilterCentroid_))));
        size_t overlapUse = shortInput ? centroidOverlap : static_cast<size_t>(overlapSize_);
        size_t inputOffset = shortInput ? 0 : overlapUse;
        int validOutputPerBlock = fftSize_ - static_cast<int>(overlapUse);

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
                cudaMemsetAsync(d_paddedInput, 0, fftSize_ * sizeof(Sample), stream),
                "cudaMemset padded input"
            );

            // Copy overlap from previous block (host to device)
            if (overlapUse > 0 && outputPos > 0) {
                Utils::checkCudaError(
                    copyHostToDeviceSamplesAsync(d_paddedInput, overlapBuffer.data(),
                                                 overlapUse, stream),
                    "cudaMemcpy overlap to device"
                );

                if (blockCount < 3) {
                    fprintf(stderr, "[DEBUG] Block %zu: Loaded overlap - first sample=%.6f, last sample=%.6f\n",
                            blockCount, overlapBuffer[0], overlapBuffer[overlapUse-1]);
                }
            }

            // Copy new input data from upsampled signal
            if (inputPos + currentBlockSize <= outputFrames) {
                Utils::checkCudaError(
                    cudaMemcpyAsync(d_paddedInput + inputOffset, d_upsampledInput + inputPos,
                                   currentBlockSize * sizeof(Sample), cudaMemcpyDeviceToDevice,
                                   stream),
                    "cudaMemcpy block to padded"
                );
            }

            // Perform FFT convolution on this block
            Utils::checkCufftError(
                cufftSetStream(fftPlanForward_, stream),
                "cufftSetStream forward (offline)"
            );
            Utils::checkCufftError(
                Precision::execForward(fftPlanForward_, d_paddedInput, d_inputFFT),
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
                Precision::execInverse(fftPlanInverse_, d_inputFFT, d_convResult),
                "cufftExecC2R block"
            );

            // Scale by FFT size
            ScaleType scale = Precision::scaleFactor(fftSize_);
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
                downconvertToHost(outputData.data() + outputPos,
                                  d_convResult + overlapUse,
                                  validOutputSize, stream),
                "downconvert output to host"
            );

            // Save overlap for next block using the contiguous upsampled input
            if (overlapUse > 0) {
                size_t nextBlockStart = inputPos + validOutputSize;
                if (nextBlockStart >= overlapUse && nextBlockStart <= outputFrames) {
                    size_t overlapStart = nextBlockStart - overlapUse;
                    if (overlapStart + overlapUse <= outputFrames) {
                        Utils::checkCudaError(
                            downconvertToHost(overlapBuffer.data(),
                                              d_upsampledInput + overlapStart,
                                              overlapUse, stream),
                            "downconvert overlap from device"
                        );
                    }
                }
            }

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

int GPUUpsampler::getPhaseCrossfadeSamples() const {
    if (!streamInitialized_) {
        return 0;
    }
    constexpr int kCrossfadeMs = 12;
    int outputRate = inputSampleRate_ * upsampleRatio_;
    if (outputRate <= 0) {
        return 0;
    }
    int samples = static_cast<int>((static_cast<long double>(outputRate) * kCrossfadeMs) / 1000.0L);
    if (samples <= 0) {
        return 0;
    }
    if (validOutputPerBlock_ > 0) {
        samples = std::max(samples, validOutputPerBlock_);
    }
    return samples;
}

float GPUUpsampler::getCurrentGroupDelay() const {
    if (multiRateEnabled_ && currentMultiRateIndex_ >= 0 &&
        currentMultiRateIndex_ < MULTI_RATE_CONFIG_COUNT) {
        return filterCentroidMulti_[currentMultiRateIndex_];
    }
    if (quadPhaseEnabled_) {
        if (currentRateFamily_ == RateFamily::RATE_44K) {
            return (phaseType_ == PhaseType::Minimum) ? filterCentroid44k_ : filterCentroid44kLinear_;
        }
        return (phaseType_ == PhaseType::Minimum) ? filterCentroid48k_ : filterCentroid48kLinear_;
    }
    if (dualRateEnabled_) {
        return (currentRateFamily_ == RateFamily::RATE_44K) ? filterCentroid44k_ : filterCentroid48k_;
    }
    return baseFilterCentroid_;
}

void GPUUpsampler::cancelPhaseAlignedCrossfade() {
    phaseCrossfade_.active = false;
    phaseCrossfade_.previousFilter = nullptr;
    phaseCrossfade_.samplesRemaining = 0;
    phaseCrossfade_.totalSamples = 0;
    phaseCrossfade_.samplesProcessed = 0;
    phaseCrossfade_.previousDelay = 0.0f;
    phaseCrossfade_.newDelay = 0.0f;
    phaseCrossfade_.delayOld.reset();
    phaseCrossfade_.delayNewLine.reset();
    crossfadeOldOutput_.clear();
    crossfadeAlignedOld_.clear();
    crossfadeAlignedNew_.clear();
}

void GPUUpsampler::startPhaseAlignedCrossfade(Complex* previousFilter,
                                              float previousDelay,
                                              float newDelay) {
    if (!streamInitialized_ || filterFftSize_ == 0 || previousFilter == nullptr) {
        return;
    }
    int crossfadeSamples = getPhaseCrossfadeSamples();
    if (crossfadeSamples <= 0) {
        return;
    }

    size_t fftBytes = filterFftSize_ * sizeof(Complex);
    if (!d_crossfadeFilterSnapshot_) {
        Utils::checkCudaError(
            cudaMalloc(&d_crossfadeFilterSnapshot_, fftBytes),
            "cudaMalloc crossfade snapshot"
        );
    }

    Utils::checkCudaError(
        cudaMemcpy(d_crossfadeFilterSnapshot_, previousFilter, fftBytes, cudaMemcpyDeviceToDevice),
        "cudaMemcpy crossfade snapshot"
    );

    phaseCrossfade_.active = true;
    phaseCrossfade_.previousFilter = d_crossfadeFilterSnapshot_;
    phaseCrossfade_.previousDelay = previousDelay;
    phaseCrossfade_.newDelay = newDelay;
    phaseCrossfade_.samplesRemaining = crossfadeSamples;
    phaseCrossfade_.totalSamples = crossfadeSamples;
    phaseCrossfade_.samplesProcessed = 0;

    float delta = previousDelay - newDelay;
    float absDelta = std::fabs(delta);
    if (absDelta < 1e-4f) {
        phaseCrossfade_.delayNewLine.configure(0.0f);
        phaseCrossfade_.delayOld.configure(0.0f);
    } else if (delta > 0.0f) {
        phaseCrossfade_.delayNew = true;
        phaseCrossfade_.delayNewLine.configure(delta);
        phaseCrossfade_.delayOld.configure(0.0f);
    } else {
        phaseCrossfade_.delayNew = false;
        phaseCrossfade_.delayOld.configure(-delta);
        phaseCrossfade_.delayNewLine.configure(0.0f);
    }

    phaseCrossfade_.delayOld.reset();
    phaseCrossfade_.delayNewLine.reset();
}

void GPUUpsampler::applyPhaseAlignedCrossfade(StreamFloatVector& newOutput,
                                              const std::vector<float>& oldOutput,
                                              bool advanceProgress) {
    if (!phaseCrossfade_.active) {
        return;
    }

    size_t frames = std::min(newOutput.size(), oldOutput.size());
    if (frames == 0) {
        return;
    }

    const std::vector<float>* oldPtr = &oldOutput;
    const std::vector<float>* newPtr = &crossfadeAlignedNew_;

    if (!phaseCrossfade_.delayOld.isBypassed()) {
        phaseCrossfade_.delayOld.process(oldOutput, crossfadeAlignedOld_);
        oldPtr = &crossfadeAlignedOld_;
    } else {
        crossfadeAlignedOld_.clear();
    }

    if (!phaseCrossfade_.delayNewLine.isBypassed()) {
        phaseCrossfade_.delayNewLine.process(newOutput, crossfadeAlignedNew_);
    } else {
        crossfadeAlignedNew_.assign(newOutput.begin(), newOutput.end());
    }

    float denom = static_cast<float>(std::max(phaseCrossfade_.totalSamples, 1));
    for (size_t i = 0; i < frames; ++i) {
        float progress = static_cast<float>(phaseCrossfade_.samplesProcessed + i) / denom;
        progress = std::clamp(progress, 0.0f, 1.0f);
        float gainNew = progress;
        float gainOld = 1.0f - progress;
        newOutput[i] = (*oldPtr)[i] * gainOld + (*newPtr)[i] * gainNew;
    }

    if (advanceProgress) {
        phaseCrossfade_.samplesProcessed += static_cast<int>(frames);
        if (phaseCrossfade_.samplesProcessed >= phaseCrossfade_.totalSamples) {
            cancelPhaseAlignedCrossfade();
        } else {
            phaseCrossfade_.samplesRemaining =
                phaseCrossfade_.totalSamples - phaseCrossfade_.samplesProcessed;
        }
    }
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
    if (d_outputScratch_) cudaFree(d_outputScratch_);
    if (d_outputScratchLeft_) cudaFree(d_outputScratchLeft_);
    if (d_outputScratchRight_) cudaFree(d_outputScratchRight_);
    if (d_crossfadeFilterSnapshot_) cudaFree(d_crossfadeFilterSnapshot_);

    // Free streaming buffers
    if (d_streamInput_) cudaFree(d_streamInput_);
    if (d_streamUpsampled_) cudaFree(d_streamUpsampled_);
    if (d_streamPadded_) cudaFree(d_streamPadded_);
    if (d_streamInputFFT_) cudaFree(d_streamInputFFT_);
    if (d_streamConvResult_) cudaFree(d_streamConvResult_);
    if (d_streamInputFFTBackup_) cudaFree(d_streamInputFFTBackup_);
    if (d_streamConvResultOld_) cudaFree(d_streamConvResultOld_);
    if (d_overlapLeft_) cudaFree(d_overlapLeft_);
    if (d_overlapRight_) cudaFree(d_overlapRight_);
    if (d_tailAccumulator_) cudaFree(d_tailAccumulator_);
    if (d_tailMixBuffer_) cudaFree(d_tailMixBuffer_);
    if (d_upsampledHistory_) cudaFree(d_upsampledHistory_);

    if (fftPlanForward_) cufftDestroy(fftPlanForward_);
    if (fftPlanInverse_) cufftDestroy(fftPlanInverse_);
    if (partitionImpulsePlanInverse_) {
        cufftDestroy(partitionImpulsePlanInverse_);
        partitionImpulsePlanInverse_ = 0;
    }

    // Free EQ-specific resources
    if (eqPlanD2Z_) cufftDestroy(eqPlanD2Z_);
    if (eqPlanZ2D_) cufftDestroy(eqPlanZ2D_);
    if (d_eqLogMag_) cudaFree(d_eqLogMag_);
    if (d_eqComplexSpec_) cudaFree(d_eqComplexSpec_);
    if (d_partitionImpulse_) {
        cudaFree(d_partitionImpulse_);
        d_partitionImpulse_ = nullptr;
    }
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
    d_crossfadeFilterSnapshot_ = nullptr;
    filterFftSize_ = 0;
    eqApplied_ = false;
    d_inputBlock_ = nullptr;
    d_outputBlock_ = nullptr;
    d_inputFFT_ = nullptr;
    d_convResult_ = nullptr;
    d_outputScratch_ = nullptr;
    d_outputScratchLeft_ = nullptr;
    d_outputScratchRight_ = nullptr;
    d_streamInput_ = nullptr;
    d_streamUpsampled_ = nullptr;
    d_streamPadded_ = nullptr;
    d_streamInputFFT_ = nullptr;
    d_streamConvResult_ = nullptr;
    d_streamInputFFTBackup_ = nullptr;
    d_streamConvResultOld_ = nullptr;
    d_overlapLeft_ = nullptr;
    d_overlapRight_ = nullptr;
    d_tailAccumulator_ = nullptr;
    d_tailMixBuffer_ = nullptr;
    d_upsampledHistory_ = nullptr;
    fftPlanForward_ = 0;
    fftPlanInverse_ = 0;
    stream_ = nullptr;
    streamLeft_ = nullptr;
    streamRight_ = nullptr;
    streamValidInputPerBlock_ = 0;
    validOutputPerBlock_ = 0;
    streamOverlapSize_ = 0;
    streamInitialized_ = false;

    freePartitionStates();
}

void GPUUpsampler::resetPartitionedStreaming() {
    if (!partitionStreamingInitialized_) {
        return;
    }

    if (d_overlapLeft_) {
        Utils::checkCudaError(
            cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(Sample)),
            "cudaMemset partition reset overlap left");
    }
    if (d_overlapRight_) {
        Utils::checkCudaError(
            cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(Sample)),
            "cudaMemset partition reset overlap right");
    }

    for (auto& state : partitionStates_) {
        if (state.d_overlapLeft) {
            Utils::checkCudaError(
                cudaMemset(state.d_overlapLeft, 0, state.overlapSize * sizeof(Sample)),
                "cudaMemset partition state reset overlap left");
        }
        if (state.d_overlapRight) {
            Utils::checkCudaError(
                cudaMemset(state.d_overlapRight, 0, state.overlapSize * sizeof(Sample)),
                "cudaMemset partition state reset overlap right");
        }
    }

    tailBaseSample_ = 0;
    tailBaseIndex_ = 0;
    partitionProcessedSamples_ = 0;
    partitionOutputSamples_ = 0;
    historyWriteIndex_ = 0;
}

void GPUUpsampler::releaseHostCoefficients() {
    if (partitionConfig_.enabled) {
        std::cout << "  Partition mode active: retaining host coefficient buffers for low-latency path"
                  << std::endl;
        return;
    }
    // Release all CPU-side filter coefficient vectors to free memory
    // This is called after GPU transfer is complete and FFT spectra are on GPU
    // Important for Jetson Unified Memory optimization (saves ~100MB)

    size_t freedBytes = 0;

    // Single-rate coefficients
    if (!h_filterCoeffs_.empty()) {
        freedBytes += h_filterCoeffs_.capacity() * sizeof(float);
        h_filterCoeffs_.clear();
        h_filterCoeffs_.shrink_to_fit();
    }
    if (!h_filterCoeffsTyped_.empty()) {
        freedBytes += h_filterCoeffsTyped_.capacity() * sizeof(Sample);
        h_filterCoeffsTyped_.clear();
        h_filterCoeffsTyped_.shrink_to_fit();
    }

    // Dual-rate coefficients
    if (!h_filterCoeffs44k_.empty()) {
        freedBytes += h_filterCoeffs44k_.capacity() * sizeof(float);
        h_filterCoeffs44k_.clear();
        h_filterCoeffs44k_.shrink_to_fit();
    }
    if (!h_filterCoeffs48k_.empty()) {
        freedBytes += h_filterCoeffs48k_.capacity() * sizeof(float);
        h_filterCoeffs48k_.clear();
        h_filterCoeffs48k_.shrink_to_fit();
    }

    // Quad-phase (linear) coefficients
    if (!h_filterCoeffs44k_linear_.empty()) {
        freedBytes += h_filterCoeffs44k_linear_.capacity() * sizeof(float);
        h_filterCoeffs44k_linear_.clear();
        h_filterCoeffs44k_linear_.shrink_to_fit();
    }
    if (!h_filterCoeffs48k_linear_.empty()) {
        freedBytes += h_filterCoeffs48k_linear_.capacity() * sizeof(float);
        h_filterCoeffs48k_linear_.clear();
        h_filterCoeffs48k_linear_.shrink_to_fit();
    }

    // Multi-rate coefficients (8 configurations)
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        if (!h_filterCoeffsMulti_[i].empty()) {
            freedBytes += h_filterCoeffsMulti_[i].capacity() * sizeof(float);
            h_filterCoeffsMulti_[i].clear();
            h_filterCoeffsMulti_[i].shrink_to_fit();
        }
    }

    if (freedBytes > 0) {
        std::cout << "  Released CPU coefficient memory: "
                  << (freedBytes / (1024 * 1024)) << " MB ("
                  << freedBytes << " bytes)" << std::endl;
    }
}

bool GPUUpsampler::setupPartitionStates() {
    freePartitionStates();

    if (!partitionPlan_.enabled) {
        return true;
    }

    if (h_filterCoeffs_.empty()) {
        std::cerr << "[Partition] Host filter coefficients unavailable; disabling partition mode"
                  << std::endl;
        partitionPlan_ = PartitionPlan{};
        return false;
    }

    partitionStates_.reserve(partitionPlan_.partitions.size());
    size_t coeffOffset = 0;
    maxPartitionValidOutput_ = 0;

    for (size_t idx = 0; idx < partitionPlan_.partitions.size(); ++idx) {
        const auto& descriptor = partitionPlan_.partitions[idx];
        PartitionState state;
        state.descriptor = descriptor;
        state.validOutput = descriptor.validOutput;
        state.overlapSize = std::max(0, descriptor.taps - 1);
        state.fftComplexSize = descriptor.fftSize / 2 + 1;
        state.sampleOffset = static_cast<int64_t>(coeffOffset);

        if (descriptor.taps <= 0 || descriptor.fftSize <= 0 || state.validOutput <= 0) {
            std::cerr << "[Partition] Invalid descriptor at index " << idx << std::endl;
            freePartitionStates();
            partitionPlan_ = PartitionPlan{};
            return false;
        }

        if (coeffOffset + descriptor.taps > h_filterCoeffs_.size()) {
            std::cerr << "[Partition] Descriptor taps exceed available coefficients (idx=" << idx
                      << ")" << std::endl;
            freePartitionStates();
            partitionPlan_ = PartitionPlan{};
            return false;
        }

        // Allocate frequency-domain storage for this partition's filter
        for (int bufIdx = 0; bufIdx < 2; ++bufIdx) {
            Utils::checkCudaError(
                cudaMalloc(&state.d_filterFFT[bufIdx],
                           state.fftComplexSize * sizeof(Complex)),
                "cudaMalloc partition filter FFT buffer");
        }

        Sample* d_tempTime = nullptr;
        Utils::checkCudaError(
            cudaMalloc(&d_tempTime, descriptor.fftSize * sizeof(Sample)),
            "cudaMalloc partition temp buffer");
        Utils::checkCudaError(
            cudaMemset(d_tempTime, 0, descriptor.fftSize * sizeof(Sample)),
            "cudaMemset partition temp buffer");

        // Copy the tap segment to device (remaining samples stay zero-padded)
        Utils::checkCudaError(
            copyHostToDeviceSamples<Precision>(d_tempTime, h_filterCoeffs_.data() + coeffOffset,
                                               descriptor.taps),
            "cudaMemcpy partition taps");

        cufftHandle plan = 0;
        Utils::checkCufftError(
            cufftPlan1d(&plan, descriptor.fftSize, Precision::kFftTypeForward, 1),
            "cufftPlan1d partition filter");
        Utils::checkCufftError(
            Precision::execForward(plan, d_tempTime, state.d_filterFFT[0]),
            "cufftExecR2C partition filter");
        cufftDestroy(plan);
        cudaFree(d_tempTime);
        Utils::checkCudaError(
            cudaMemcpy(state.d_filterFFT[1], state.d_filterFFT[0],
                       state.fftComplexSize * sizeof(Complex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy partition filter FFT mirror");
        state.activeFilterIndex = 0;
        coeffOffset += descriptor.taps;
        maxPartitionValidOutput_ =
            std::max(maxPartitionValidOutput_, static_cast<size_t>(state.validOutput));

        partitionStates_.push_back(state);
    }

    partitionFastIndex_ = 0;

    std::cout << "[Partition] Prepared " << partitionStates_.size()
              << " partition state(s) for streaming" << std::endl;
    return true;
}

void GPUUpsampler::freePartitionStates() {
    for (auto& state : partitionStates_) {
        for (int bufIdx = 0; bufIdx < 2; ++bufIdx) {
            if (state.d_filterFFT[bufIdx]) {
                cudaFree(state.d_filterFFT[bufIdx]);
                state.d_filterFFT[bufIdx] = nullptr;
            }
        }
        if (state.d_timeDomain) {
            cudaFree(state.d_timeDomain);
            state.d_timeDomain = nullptr;
        }
        if (state.d_inputFFT) {
            cudaFree(state.d_inputFFT);
            state.d_inputFFT = nullptr;
        }
        if (state.d_overlapLeft) {
            cudaFree(state.d_overlapLeft);
            state.d_overlapLeft = nullptr;
        }
        if (state.d_overlapRight) {
            cudaFree(state.d_overlapRight);
            state.d_overlapRight = nullptr;
        }
        if (state.planForward) {
            cufftDestroy(state.planForward);
            state.planForward = 0;
        }
        if (state.planInverse) {
            cufftDestroy(state.planInverse);
            state.planInverse = 0;
        }
    }
    partitionStates_.clear();
    partitionStreamingInitialized_ = false;
    partitionFastFftSize_ = 0;
    partitionFastFftComplexSize_ = 0;
    maxPartitionValidOutput_ = 0;
}

bool GPUUpsampler::initializePartitionedStreaming() {
    if (!partitionPlan_.enabled) {
        return initializeStreaming();
    }

    if (partitionStates_.empty() && !setupPartitionStates()) {
        return false;
    }

    if (partitionStates_.empty()) {
        std::cerr << "[Partition] No partition states available" << std::endl;
        return false;
    }

    const auto& fastState = partitionStates_.front();
    int fastValid = fastState.descriptor.fftSize - fastState.overlapSize;
    int adjustedValid = (fastValid / upsampleRatio_) * upsampleRatio_;
    if (adjustedValid <= 0) {
        adjustedValid = fastValid;
    }

    int blockSamples = adjustedValid;
    for (auto& state : partitionStates_) {
        int maxValid = state.descriptor.fftSize - state.overlapSize;
        if (maxValid <= 0) {
            std::cerr << "[Partition] Descriptor FFT too small for overlap (taps="
                      << state.descriptor.taps << ", fft=" << state.descriptor.fftSize << ")"
                      << std::endl;
            return false;
        }
        state.validOutput = std::min(adjustedValid, maxValid);
        blockSamples = std::min(blockSamples, state.validOutput);
    }

    int fastInput = blockSamples / upsampleRatio_;
    if (fastInput <= 0) {
        std::cerr << "[Partition] Invalid fast partition configuration (input=0)" << std::endl;
        return false;
    }

    blockSamples = fastInput * upsampleRatio_;
    if (blockSamples <= 0) {
        blockSamples = adjustedValid;
    }

    // Free existing streaming buffers before reallocating
    freeStreamingBuffers();

    streamOverlapSize_ = fastState.overlapSize;
    validOutputPerBlock_ = blockSamples;
    streamValidInputPerBlock_ = fastInput;
    partitionFastFftSize_ = fastState.descriptor.fftSize;
    partitionFastFftComplexSize_ = fastState.fftComplexSize;

    size_t upsampledSize = static_cast<size_t>(streamValidInputPerBlock_) * upsampleRatio_;

    Utils::checkCudaError(
        cudaMalloc(&d_streamInput_, streamValidInputPerBlock_ * sizeof(Sample)),
        "cudaMalloc partition streaming input");
    Utils::checkCudaError(
        cudaMalloc(&d_streamUpsampled_, upsampledSize * sizeof(Sample)),
        "cudaMalloc partition streaming upsampled");
    Utils::checkCudaError(
        cudaMalloc(&d_streamPadded_, partitionFastFftSize_ * sizeof(Sample)),
        "cudaMalloc partition streaming padded");
    Utils::checkCudaError(
        cudaMalloc(&d_streamInputFFT_, partitionFastFftComplexSize_ * sizeof(Complex)),
        "cudaMalloc partition streaming FFT");
    Utils::checkCudaError(
        cudaMalloc(&d_streamInputFFTBackup_, partitionFastFftComplexSize_ * sizeof(Complex)),
        "cudaMalloc partition streaming FFT backup");
    Utils::checkCudaError(
        cudaMalloc(&d_streamConvResult_, partitionFastFftSize_ * sizeof(Sample)),
        "cudaMalloc partition streaming conv result");
    Utils::checkCudaError(
        cudaMalloc(&d_streamConvResultOld_, partitionFastFftSize_ * sizeof(Sample)),
        "cudaMalloc partition streaming old conv result");

    Utils::checkCudaError(
        cudaMalloc(&d_overlapLeft_, streamOverlapSize_ * sizeof(Sample)),
        "cudaMalloc partition overlap left");
    Utils::checkCudaError(
        cudaMalloc(&d_overlapRight_, streamOverlapSize_ * sizeof(Sample)),
        "cudaMalloc partition overlap right");
    Utils::checkCudaError(
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(Sample)),
        "cudaMemset partition overlap left");
    Utils::checkCudaError(
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(Sample)),
        "cudaMemset partition overlap right");

    // Allocate runtime buffers for each partition
    for (size_t idx = 0; idx < partitionStates_.size(); ++idx) {
        auto& state = partitionStates_[idx];

        if (state.d_timeDomain) {
            cudaFree(state.d_timeDomain);
            state.d_timeDomain = nullptr;
        }
        if (state.d_inputFFT) {
            cudaFree(state.d_inputFFT);
            state.d_inputFFT = nullptr;
        }
        if (state.d_overlapLeft) {
            cudaFree(state.d_overlapLeft);
            state.d_overlapLeft = nullptr;
        }
        if (state.d_overlapRight) {
            cudaFree(state.d_overlapRight);
            state.d_overlapRight = nullptr;
        }
        if (state.planForward) {
            cufftDestroy(state.planForward);
            state.planForward = 0;
        }
        if (state.planInverse) {
            cufftDestroy(state.planInverse);
            state.planInverse = 0;
        }

        Utils::checkCudaError(
            cudaMalloc(&state.d_timeDomain, state.descriptor.fftSize * sizeof(Sample)),
            "cudaMalloc partition time buffer");
        Utils::checkCudaError(
            cudaMalloc(&state.d_inputFFT, state.fftComplexSize * sizeof(Complex)),
            "cudaMalloc partition input FFT");

        if (state.overlapSize > 0) {
            Utils::checkCudaError(
                cudaMalloc(&state.d_overlapLeft, state.overlapSize * sizeof(Sample)),
                "cudaMalloc partition overlap left");
            Utils::checkCudaError(
                cudaMalloc(&state.d_overlapRight, state.overlapSize * sizeof(Sample)),
                "cudaMalloc partition overlap right");
            Utils::checkCudaError(
                cudaMemset(state.d_overlapLeft, 0, state.overlapSize * sizeof(Sample)),
                "cudaMemset partition state overlap left");
            Utils::checkCudaError(
                cudaMemset(state.d_overlapRight, 0, state.overlapSize * sizeof(Sample)),
                "cudaMemset partition state overlap right");
        }

        Utils::checkCufftError(
            cufftPlan1d(&state.planForward, state.descriptor.fftSize, Precision::kFftTypeForward, 1),
            "cufftPlan1d partition forward");
        Utils::checkCufftError(
            cufftPlan1d(&state.planInverse, state.descriptor.fftSize, Precision::kFftTypeInverse, 1),
            "cufftPlan1d partition inverse");
    }

    streamInitialized_ = true;
    partitionStreamingInitialized_ = true;

    std::cout << "[Partition] Streaming initialized (fast FFT " << partitionFastFftSize_
              << ", valid output " << validOutputPerBlock_
              << ", input samples per block " << streamValidInputPerBlock_ << ")" << std::endl;
    return true;
}

bool GPUUpsampler::processPartitionBlock(PartitionState& state, cudaStream_t stream,
                                         const Sample* d_newSamples, int newSamples,
                                         Sample* d_channelOverlap, StreamFloatVector& tempOutput,
                                         StreamFloatVector& outputData) {
    try {
        int samplesToUse = std::min(newSamples, state.validOutput);
        if (samplesToUse <= 0) {
            return true;
        }

        Utils::checkCudaError(
            cudaMemsetAsync(state.d_timeDomain, 0, state.descriptor.fftSize * sizeof(Sample),
                            stream),
            "cudaMemset partition time buffer");

        if (state.overlapSize > 0 && d_channelOverlap) {
            Utils::checkCudaError(
                cudaMemcpyAsync(state.d_timeDomain, d_channelOverlap,
                                state.overlapSize * sizeof(Sample),
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy partition overlap prepend");
        }

        Utils::checkCudaError(
            cudaMemcpyAsync(state.d_timeDomain + state.overlapSize, d_newSamples,
                            samplesToUse * sizeof(Sample), cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy partition block");

        if (state.overlapSize > 0 && d_channelOverlap) {
            int overlapOffset = std::max(0, samplesToUse);
            size_t maxCopy =
                static_cast<size_t>(state.descriptor.fftSize) - static_cast<size_t>(overlapOffset);
            size_t overlapSamples =
                std::min(maxCopy, static_cast<size_t>(state.overlapSize));
            if (overlapSamples > 0) {
                Utils::checkCudaError(
                    cudaMemcpyAsync(d_channelOverlap, state.d_timeDomain + overlapOffset,
                                    overlapSamples * sizeof(Sample),
                                    cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy partition overlap save");
            }
        }

        Utils::checkCufftError(cufftSetStream(state.planForward, stream),
                               "cufftSetStream partition forward");
        Utils::checkCufftError(
            Precision::execForward(state.planForward, state.d_timeDomain, state.d_inputFFT),
            "cufftExecR2C partition block");

        int threadsPerBlock = 256;
        int blocks = (state.fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
        Complex* activeFilter = state.d_filterFFT[state.activeFilterIndex];
        complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            state.d_inputFFT, activeFilter, state.fftComplexSize);

        Utils::checkCufftError(cufftSetStream(state.planInverse, stream),
                               "cufftSetStream partition inverse");
        Utils::checkCufftError(
            Precision::execInverse(state.planInverse, state.d_inputFFT, state.d_timeDomain),
            "cufftExecC2R partition block");

        ScaleType scale = Precision::scaleFactor(state.descriptor.fftSize);
        int scaleBlocks = (state.descriptor.fftSize + threadsPerBlock - 1) / threadsPerBlock;
        scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
            state.d_timeDomain, state.descriptor.fftSize, scale);

        tempOutput.resize(samplesToUse);
        registerStreamOutputBuffer(tempOutput, stream);
        Utils::checkCudaError(
            downconvertToHost(tempOutput.data(), state.d_timeDomain + state.overlapSize,
                              samplesToUse, stream),
            "downconvert partition output");
        Utils::checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize partition");

        if (outputData.size() < static_cast<size_t>(samplesToUse)) {
            outputData.resize(samplesToUse);
        }
        for (int i = 0; i < samplesToUse; ++i) {
            outputData[i] += tempOutput[i];
        }
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Partition] Error: " << e.what() << std::endl;
        return false;
    }
}

void GPUUpsampler::setActiveHostCoefficients(const std::vector<float>& source) {
    size_t desiredTaps = static_cast<size_t>(std::max(filterTaps_, static_cast<int>(source.size())));
    if (desiredTaps == 0) {
        desiredTaps = source.size();
        filterTaps_ = static_cast<int>(desiredTaps);
    }
    h_filterCoeffs_.assign(desiredTaps, 0.0f);
    const size_t copyCount = std::min(source.size(), h_filterCoeffs_.size());
    if (copyCount > 0) {
        std::copy(source.begin(), source.begin() + copyCount, h_filterCoeffs_.begin());
    }
}

bool GPUUpsampler::updateActiveImpulseFromSpectrum(const Complex* spectrum,
                                                   std::vector<float>& destination) {
    if (!spectrum || fftSize_ <= 0) {
        return false;
    }
    if (filterTaps_ <= 0) {
        return false;
    }
    if (!partitionImpulsePlanInverse_) {
        Utils::checkCufftError(
            cufftPlan1d(&partitionImpulsePlanInverse_, fftSize_, Precision::kFftTypeInverse, 1),
            "cufftPlan1d partition impulse inverse (lazy)");
    }
    size_t impulseBytes = static_cast<size_t>(fftSize_) * sizeof(Sample);
    if (!d_partitionImpulse_) {
        Utils::checkCudaError(
            cudaMalloc(&d_partitionImpulse_, impulseBytes),
            "cudaMalloc partition impulse buffer");
    }

    Utils::checkCufftError(
        Precision::execInverse(partitionImpulsePlanInverse_, const_cast<Complex*>(spectrum),
                               d_partitionImpulse_),
        "cufftExecC2R partition impulse build");
    Utils::checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize partition impulse build");

    destination.assign(static_cast<size_t>(filterTaps_), 0.0f);
    Utils::checkCudaError(
        downconvertToHostSync(destination.data(), d_partitionImpulse_,
                              static_cast<size_t>(filterTaps_)),
        "downconvert partition impulse to host");
    const float scale = static_cast<float>(Precision::scaleFactor(fftSize_));
    for (auto& sample : destination) {
        sample *= scale;
    }
    return true;
}

bool GPUUpsampler::refreshPartitionFiltersFromHost() {
    if (!partitionPlan_.enabled || partitionStates_.empty()) {
        return true;
    }
    if (h_filterCoeffs_.empty()) {
        std::cerr << "[Partition] Host coefficients unavailable; cannot refresh spectra"
                  << std::endl;
        return false;
    }

    int maxFft = 0;
    for (const auto& state : partitionStates_) {
        maxFft = std::max(maxFft, state.descriptor.fftSize);
    }
    Sample* d_tempTime = nullptr;
    Utils::checkCudaError(
        cudaMalloc(&d_tempTime, static_cast<size_t>(maxFft) * sizeof(Sample)),
        "cudaMalloc partition refresh temp buffer");

    size_t coeffOffset = 0;
    for (auto& state : partitionStates_) {
        const size_t taps = static_cast<size_t>(state.descriptor.taps);
        if (coeffOffset + taps > h_filterCoeffs_.size()) {
            std::cerr << "[Partition] Refresh failed: tap window exceeds host coefficients"
                      << std::endl;
            cudaFree(d_tempTime);
            return false;
        }

        Utils::checkCudaError(
            cudaMemset(d_tempTime, 0, state.descriptor.fftSize * sizeof(Sample)),
            "cudaMemset partition refresh temp buffer");
        Utils::checkCudaError(
            copyHostToDeviceSamples<Precision>(d_tempTime, h_filterCoeffs_.data() + coeffOffset,
                                               taps),
            "cudaMemcpy partition refresh taps");

        cufftHandle plan = 0;
        Utils::checkCufftError(
            cufftPlan1d(&plan, state.descriptor.fftSize, Precision::kFftTypeForward, 1),
            "cufftPlan1d partition refresh");
        int inactiveIndex = 1 - state.activeFilterIndex;
        Utils::checkCufftError(
            Precision::execForward(plan, d_tempTime, state.d_filterFFT[inactiveIndex]),
            "cufftExecR2C partition refresh");
        Utils::checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize partition refresh");
        cufftDestroy(plan);

        state.activeFilterIndex = inactiveIndex;
        coeffOffset += taps;
    }

    cudaFree(d_tempTime);
    std::cout << "[Partition] Filter spectra refreshed for current impulse response" << std::endl;
    return true;
}

bool GPUUpsampler::refreshPartitionFiltersFromActiveSpectrum() {
    if (!updateActiveImpulseFromSpectrum(d_activeFilterFFT_, h_filterCoeffs_)) {
        return false;
    }
    if (!partitionPlan_.enabled) {
        return true;
    }
    return refreshPartitionFiltersFromHost();
}

}  // namespace ConvolutionEngine
