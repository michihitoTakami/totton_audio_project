#include "crossfeed_engine.h"
#include "gpu/cuda_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

// JSON parsing (simple implementation for metadata)
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace CrossfeedEngine {

// ============================================================================
// CUDA Kernels for Crossfeed
// ============================================================================

// Complex multiply-accumulate kernel
// out += in * filter (accumulates into output)
__global__ void complexMultiplyAccumulateKernel(cufftComplex* out,
                                                  const cufftComplex* in,
                                                  const cufftComplex* filter,
                                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = in[idx].x;
        float b = in[idx].y;
        float c = filter[idx].x;
        float d = filter[idx].y;
        // out += (a + bi) * (c + di)
        out[idx].x += a * c - b * d;
        out[idx].y += a * d + b * c;
    }
}

// Complex multiply and store kernel
// out = in * filter (replaces output)
__global__ void complexMultiplyStoreKernel(cufftComplex* out,
                                            const cufftComplex* in,
                                            const cufftComplex* filter,
                                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = in[idx].x;
        float b = in[idx].y;
        float c = filter[idx].x;
        float d = filter[idx].y;
        out[idx].x = a * c - b * d;
        out[idx].y = a * d + b * c;
    }
}

// Scale kernel (reuse pattern)
__global__ void scaleKernel(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace {

void checkCudaError(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << context << ": " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

void checkCufftError(cufftResult result, const char* context) {
    if (result != CUFFT_SUCCESS) {
        std::ostringstream oss;
        oss << "cuFFT error at " << context << ": " << result;
        throw std::runtime_error(oss.str());
    }
}

void safeCudaHostUnregister(void** trackedPtr, size_t* trackedBytes, const char* context) {
    if (trackedPtr == nullptr || *trackedPtr == nullptr) {
        return;
    }
    cudaError_t err = cudaHostUnregister(*trackedPtr);
    if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
        checkCudaError(err, context);
    }
    *trackedPtr = nullptr;
    if (trackedBytes) {
        *trackedBytes = 0;
    }
}

}  // namespace

void HRTFProcessor::registerStreamBuffer(std::vector<float>& buffer, void** trackedPtr,
                                         size_t* trackedBytes, const char* context) {
    if (buffer.empty()) {
        safeCudaHostUnregister(trackedPtr, trackedBytes, "cudaHostUnregister stream buffer");
        return;
    }

    void* ptr = buffer.data();
    size_t bytes = buffer.size() * sizeof(float);

    if (*trackedPtr == ptr && *trackedBytes == bytes) {
        return;  // Already registered
    }

    safeCudaHostUnregister(trackedPtr, trackedBytes, "cudaHostUnregister stream buffer");

    checkCudaError(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault), context);
    *trackedPtr = ptr;
    *trackedBytes = bytes;
}

// ============================================================================
// HRTFProcessor Implementation
// ============================================================================

HRTFProcessor::HRTFProcessor()
    : blockSize_(8192),
      filterTaps_(0),
      fftSize_(0),
      enabled_(true),
      initialized_(false),
      currentHeadSize_(HeadSize::M),
      currentRateFamily_(RateFamily::RATE_44K),
      activeFilterConfig_(-1),
      filterFftSize_(0),
      d_inputL_(nullptr),
      d_inputR_(nullptr),
      d_paddedInputL_(nullptr),
      d_paddedInputR_(nullptr),
      d_inputFFT_L_(nullptr),
      d_inputFFT_R_(nullptr),
      d_convLL_(nullptr),
      d_convLR_(nullptr),
      d_convRL_(nullptr),
      d_convRR_(nullptr),
      d_outputL_(nullptr),
      d_outputR_(nullptr),
      d_tempConv_(nullptr),
      fftPlanForward_(0),
      fftPlanInverse_(0),
      stream_(nullptr),
      streamL_(nullptr),
      streamR_(nullptr),
      overlapSize_(0),
      d_overlapL_(nullptr),
      d_overlapR_(nullptr),
      streamInitialized_(false),
      streamValidInputPerBlock_(0),
      pinnedStreamInputL_(nullptr),
      pinnedStreamInputR_(nullptr),
      pinnedStreamOutputL_(nullptr),
      pinnedStreamOutputR_(nullptr),
      pinnedStreamInputLBytes_(0),
      pinnedStreamInputRBytes_(0),
      pinnedStreamOutputLBytes_(0),
      pinnedStreamOutputRBytes_(0),
      validOutputPerBlock_(0) {
    stats_ = Stats();
    // Initialize all filter FFT pointers to nullptr
    for (int i = 0; i < NUM_CONFIGS; ++i) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            d_filterFFT_[i][c] = nullptr;
        }
    }
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        d_activeFilterFFT_[c] = nullptr;
    }
    // Initialize combined filter pointers
    for (int f = 0; f < NUM_RATE_FAMILIES; ++f) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            d_combinedFilterFFT_[f][c] = nullptr;
        }
        combinedFilterLoaded_[f] = false;
    }
    usingCombinedFilter_ = false;
}

HRTFProcessor::~HRTFProcessor() {
    cleanup();
}

bool HRTFProcessor::initialize(const std::string& hrtfDir, int blockSize,
                                HeadSize initialSize, RateFamily initialFamily) {
    blockSize_ = blockSize;
    currentHeadSize_ = initialSize;
    currentRateFamily_ = initialFamily;

    std::cout << "Initializing HRTF Processor..." << std::endl;
    std::cout << "  HRTF directory: " << hrtfDir << std::endl;
    std::cout << "  Block size: " << blockSize_ << " samples" << std::endl;

    // Load all HRTF configurations (4 sizes * 2 rate families)
    bool anyLoaded = false;
    bool tapMismatchDetected = false;
    int expectedTapCount = -1;
    for (int sizeIdx = 0; sizeIdx < static_cast<int>(HeadSize::COUNT); ++sizeIdx) {
        for (int familyIdx = 0; familyIdx < 2; ++familyIdx) {
            HeadSize size = static_cast<HeadSize>(sizeIdx);
            RateFamily family = static_cast<RateFamily>(familyIdx);

            std::string sizeStr = headSizeToString(size);
            std::string familyStr = rateFamilyToString(family);

            std::string binPath = hrtfDir + "/hrtf_" + sizeStr + "_" + familyStr + ".bin";
            std::string jsonPath = hrtfDir + "/hrtf_" + sizeStr + "_" + familyStr + ".json";

            if (!std::filesystem::exists(binPath) || !std::filesystem::exists(jsonPath)) {
                continue;  // Missing configuration is tolerated; skip silently
            }

            if (loadHRTFCoefficients(binPath, jsonPath, size, family, expectedTapCount)) {
                anyLoaded = true;
                if (expectedTapCount < 0) {
                    expectedTapCount = filterTaps_;
                }
                std::cout << "  Loaded: " << sizeStr << "/" << familyStr << std::endl;
            } else {
                tapMismatchDetected = true;
            }
        }
    }

    if (!anyLoaded) {
        std::cerr << "Error: No HRTF files loaded from " << hrtfDir << std::endl;
        return false;
    }
    if (tapMismatchDetected) {
        std::cerr << "Error: HRTF tap count mismatch across rate families or sizes. "
                  << "Ensure all HRTF filters share the same n_taps." << std::endl;
        return false;
    }

    // Setup GPU resources
    if (!setupGPUResources()) {
        return false;
    }

    // Set initial active filter
    int initialConfig = getFilterIndex(initialSize, initialFamily);
    if (d_filterFFT_[initialConfig][0] == nullptr) {
        std::cerr << "Error: Initial HRTF config not available" << std::endl;
        return false;
    }
    activeFilterConfig_ = initialConfig;
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        d_activeFilterFFT_[c] = d_filterFFT_[initialConfig][c];
    }

    // Release CPU-side coefficient memory after GPU transfer (Jetson optimization)
    // FFT spectra are now on GPU; time-domain coefficients are no longer needed
    releaseHostCoefficients();

    initialized_ = true;
    std::cout << "HRTF Processor initialized successfully!" << std::endl;
    return true;
}

bool HRTFProcessor::loadHRTFCoefficients(const std::string& binPath,
                                          const std::string& jsonPath,
                                          HeadSize size, RateFamily family, int expectedTaps) {
    // Check if files exist
    std::ifstream binFile(binPath, std::ios::binary);
    std::ifstream jsonFile(jsonPath);
    if (!binFile || !jsonFile) {
        return false;
    }

    int configIdx = getFilterIndex(size, family);

    try {
        // Load metadata
        json meta;
        jsonFile >> meta;

        HRTFMetadata& md = metadata_[configIdx];
        md.description = meta.value("description", "");
        md.sizeCategory = meta.value("size_category", "");
        md.subjectId = meta.value("subject_id", "");
        md.sampleRate = meta.value("sample_rate", 0);
        md.rateFamily = meta.value("rate_family", "");
        md.nTaps = meta.value("n_taps", 0);
        md.nChannels = meta.value("n_channels", 4);
        md.phaseType = meta.value("phase_type", "original");
        md.normalization = meta.value("normalization", "ild_preserving");
        md.maxDcGain = meta.value("max_dc_gain", 1.0f);
        md.sourceAzimuthLeft = meta.value("source_azimuth_left", -30.0f);
        md.sourceAzimuthRight = meta.value("source_azimuth_right", 30.0f);
        md.sourceElevation = meta.value("source_elevation", 0.0f);
        md.license = meta.value("license", "");
        md.attribution = meta.value("attribution", "");
        md.source = meta.value("source", "");
        md.storageFormat = meta.value("storage_format", "");
        if (md.storageFormat.empty()) {
            md.storageFormat = "tap_interleaved_v1";  // Backward compatibility
        }

        if (meta.contains("channel_order")) {
            md.channelOrder.clear();
            for (const auto& ch : meta["channel_order"]) {
                md.channelOrder.push_back(ch.get<std::string>());
            }
        }

        if (expectedTaps > 0 && md.nTaps != expectedTaps) {
            std::cerr << "Error: HRTF tap count mismatch for " << binPath << " (expected "
                      << expectedTaps << ", got " << md.nTaps << ")" << std::endl;
            return false;
        }

        // Set filter taps from first loaded config and enforce consistency
        if (filterTaps_ == 0) {
            filterTaps_ = md.nTaps;
        } else if (filterTaps_ != md.nTaps) {
            std::cerr << "Error: HRTF tap count mismatch: expected " << filterTaps_
                      << ", got " << md.nTaps << std::endl;
            return false;
        }

        // Get file size
        binFile.seekg(0, std::ios::end);
        size_t fileSize = binFile.tellg();
        binFile.seekg(0, std::ios::beg);

        // Expected: 4 channels * nTaps * sizeof(float)
        size_t expectedSize = static_cast<size_t>(md.nChannels) * md.nTaps * sizeof(float);
        if (fileSize != expectedSize) {
            std::cerr << "Error: HRTF file size mismatch: expected " << expectedSize
                      << ", got " << fileSize << std::endl;
            return false;
        }

        size_t totalFloats = static_cast<size_t>(md.nChannels) * md.nTaps;
        std::vector<float> raw(totalFloats);
        binFile.read(reinterpret_cast<char*>(raw.data()), totalFloats * sizeof(float));
        if (!binFile) {
            std::cerr << "Error: Failed to read HRTF binary data (" << binPath << ")"
                      << std::endl;
            return false;
        }

        bool channelMajor = (md.storageFormat == "channel_major_v1");
        bool tapInterleaved = (md.storageFormat == "tap_interleaved_v1");
        if (!channelMajor && !tapInterleaved) {
            std::cerr << "Warning: Unknown HRTF storage format '" << md.storageFormat
                      << "', defaulting to tap_interleaved_v1" << std::endl;
            tapInterleaved = true;
        }

        for (int c = 0; c < NUM_CHANNELS; ++c) {
            h_filterCoeffs_[configIdx][c].resize(md.nTaps);
        }

        if (channelMajor) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                size_t offset = static_cast<size_t>(c) * md.nTaps;
                std::copy(raw.begin() + offset, raw.begin() + offset + md.nTaps,
                          h_filterCoeffs_[configIdx][c].begin());
            }
        } else {
            for (size_t tap = 0; tap < static_cast<size_t>(md.nTaps); ++tap) {
                size_t base = tap * NUM_CHANNELS;
                for (int c = 0; c < NUM_CHANNELS; ++c) {
                    h_filterCoeffs_[configIdx][c][tap] = raw[base + c];
                }
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading HRTF: " << e.what() << std::endl;
        return false;
    }
}

bool HRTFProcessor::setupGPUResources() {
    try {
        std::cout << "Setting up HRTF GPU resources..." << std::endl;

        // Create CUDA streams
        bool highPriorityMain = false;
        stream_ = ConvolutionEngine::Utils::createPrioritizedStream(
            "crossfeed cudaStreamCreate primary", cudaStreamNonBlocking, &highPriorityMain);

        bool highPriorityL = false;
        streamL_ = ConvolutionEngine::Utils::createPrioritizedStream(
            "crossfeed cudaStreamCreate left", cudaStreamNonBlocking, &highPriorityL);

        bool highPriorityR = false;
        streamR_ = ConvolutionEngine::Utils::createPrioritizedStream(
            "crossfeed cudaStreamCreate right", cudaStreamNonBlocking, &highPriorityR);

        if (highPriorityMain || highPriorityL || highPriorityR) {
            std::cout << "[Crossfeed] CUDA streams using high priority scheduling" << std::endl;
        }

        // Calculate FFT size for Overlap-Save
        size_t fftSizeNeeded = static_cast<size_t>(blockSize_) + static_cast<size_t>(filterTaps_) - 1;
        fftSize_ = 1;
        while (static_cast<size_t>(fftSize_) < fftSizeNeeded) {
            fftSize_ *= 2;
        }
        std::cout << "  FFT size: " << fftSize_ << std::endl;
        std::cout << "  Filter taps: " << filterTaps_ << std::endl;

        overlapSize_ = filterTaps_ - 1;
        overlapBufferL_.assign(overlapSize_, 0.0f);
        overlapBufferR_.assign(overlapSize_, 0.0f);

        int fftComplexSize = fftSize_ / 2 + 1;
        filterFftSize_ = fftComplexSize;

        // Allocate filter FFT for all loaded configs
        for (int configIdx = 0; configIdx < NUM_CONFIGS; ++configIdx) {
            if (h_filterCoeffs_[configIdx][0].empty()) {
                continue;  // Config not loaded
            }

            for (int c = 0; c < NUM_CHANNELS; ++c) {
                checkCudaError(
                    cudaMalloc(&d_filterFFT_[configIdx][c], fftComplexSize * sizeof(cufftComplex)),
                    "cudaMalloc filter FFT");
            }
        }

        // Allocate working buffers
        checkCudaError(cudaMalloc(&d_paddedInputL_, fftSize_ * sizeof(float)),
                       "cudaMalloc padded input L");
        checkCudaError(cudaMalloc(&d_paddedInputR_, fftSize_ * sizeof(float)),
                       "cudaMalloc padded input R");
        checkCudaError(cudaMalloc(&d_inputFFT_L_, fftComplexSize * sizeof(cufftComplex)),
                       "cudaMalloc input FFT L");
        checkCudaError(cudaMalloc(&d_inputFFT_R_, fftComplexSize * sizeof(cufftComplex)),
                       "cudaMalloc input FFT R");
        checkCudaError(cudaMalloc(&d_convLL_, fftComplexSize * sizeof(cufftComplex)),
                       "cudaMalloc conv LL");
        checkCudaError(cudaMalloc(&d_convLR_, fftComplexSize * sizeof(cufftComplex)),
                       "cudaMalloc conv LR");
        checkCudaError(cudaMalloc(&d_convRL_, fftComplexSize * sizeof(cufftComplex)),
                       "cudaMalloc conv RL");
        checkCudaError(cudaMalloc(&d_convRR_, fftComplexSize * sizeof(cufftComplex)),
                       "cudaMalloc conv RR");
        checkCudaError(cudaMalloc(&d_outputL_, fftSize_ * sizeof(float)),
                       "cudaMalloc output L");
        checkCudaError(cudaMalloc(&d_outputR_, fftSize_ * sizeof(float)),
                       "cudaMalloc output R");
        checkCudaError(cudaMalloc(&d_tempConv_, fftSize_ * sizeof(float)),
                       "cudaMalloc temp conv");

        // Device overlap buffers
        checkCudaError(cudaMalloc(&d_overlapL_, overlapSize_ * sizeof(float)),
                       "cudaMalloc device overlap L");
        checkCudaError(cudaMalloc(&d_overlapR_, overlapSize_ * sizeof(float)),
                       "cudaMalloc device overlap R");
        checkCudaError(cudaMemset(d_overlapL_, 0, overlapSize_ * sizeof(float)),
                       "cudaMemset device overlap L");
        checkCudaError(cudaMemset(d_overlapR_, 0, overlapSize_ * sizeof(float)),
                       "cudaMemset device overlap R");

        // Create cuFFT plans
        checkCufftError(cufftPlan1d(&fftPlanForward_, fftSize_, CUFFT_R2C, 1),
                        "cufftPlan1d forward");
        checkCufftError(cufftPlan1d(&fftPlanInverse_, fftSize_, CUFFT_C2R, 1),
                        "cufftPlan1d inverse");

        // Pre-compute filter FFTs
        float* d_filterPadded;
        checkCudaError(cudaMalloc(&d_filterPadded, fftSize_ * sizeof(float)),
                       "cudaMalloc filter padded");

        for (int configIdx = 0; configIdx < NUM_CONFIGS; ++configIdx) {
            if (h_filterCoeffs_[configIdx][0].empty()) {
                continue;
            }

            for (int c = 0; c < NUM_CHANNELS; ++c) {
                checkCudaError(cudaMemset(d_filterPadded, 0, fftSize_ * sizeof(float)),
                               "cudaMemset filter padded");
                size_t copyTaps =
                    std::min(static_cast<size_t>(filterTaps_), h_filterCoeffs_[configIdx][c].size());
                checkCudaError(
                    cudaMemcpy(d_filterPadded, h_filterCoeffs_[configIdx][c].data(),
                               copyTaps * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy filter to padded");
                if (copyTaps < static_cast<size_t>(filterTaps_)) {
                    size_t remaining = static_cast<size_t>(filterTaps_) - copyTaps;
                    checkCudaError(
                        cudaMemset(d_filterPadded + copyTaps, 0, remaining * sizeof(float)),
                        "cudaMemset filter padding");
                }
                checkCufftError(
                    cufftExecR2C(fftPlanForward_, d_filterPadded, d_filterFFT_[configIdx][c]),
                    "cufftExecR2C filter");
            }
        }

        cudaFree(d_filterPadded);

        // Set streaming parameters
        validOutputPerBlock_ = fftSize_ - overlapSize_;
        streamValidInputPerBlock_ = validOutputPerBlock_;  // No upsampling in crossfeed

        std::cout << "  Valid output per block: " << validOutputPerBlock_ << std::endl;
        std::cout << "  HRTF GPU resources allocated successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in setupGPUResources: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

bool HRTFProcessor::switchHeadSize(HeadSize targetSize) {
    if (!initialized_) {
        return false;
    }

    int targetConfig = getFilterIndex(targetSize, currentRateFamily_);
    if (d_filterFFT_[targetConfig][0] == nullptr) {
        std::cerr << "Error: Target HRTF config not available" << std::endl;
        return false;
    }

    if (targetConfig == activeFilterConfig_) {
        return true;  // Already at target
    }

    // Switch active filter (glitch-free: just update pointers)
    activeFilterConfig_ = targetConfig;
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        d_activeFilterFFT_[c] = d_filterFFT_[targetConfig][c];
    }
    currentHeadSize_ = targetSize;
    resetStreaming();

    std::cout << "Switched to head size: " << headSizeToString(targetSize) << std::endl;
    return true;
}

bool HRTFProcessor::switchRateFamily(RateFamily targetFamily) {
    if (!initialized_) {
        return false;
    }

    int familyIdx = static_cast<int>(targetFamily);
    int targetConfig = getFilterIndex(currentHeadSize_, targetFamily);

    // Check if combined filter is available for target family
    // Auto-restore combined filter when switching back to a family that has one loaded
    if (combinedFilterLoaded_[familyIdx]) {
        // Use combined filter (auto-restore if previously fell back to predefined)
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            d_activeFilterFFT_[c] = d_combinedFilterFFT_[familyIdx][c];
        }
        currentRateFamily_ = targetFamily;
        activeFilterConfig_ = targetConfig;
        usingCombinedFilter_ = true;
        resetStreaming();
        std::cout << "Switched to rate family: " << rateFamilyToString(targetFamily)
                  << " (using combined filter)" << std::endl;
        return true;
    }

    // Fall back to predefined filter (no combined filter for this family)
    if (d_filterFFT_[targetConfig][0] == nullptr) {
        std::cerr << "Error: Target HRTF config not available" << std::endl;
        return false;
    }

    if (targetConfig == activeFilterConfig_ && !usingCombinedFilter_) {
        return true;
    }

    activeFilterConfig_ = targetConfig;
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        d_activeFilterFFT_[c] = d_filterFFT_[targetConfig][c];
    }
    currentRateFamily_ = targetFamily;
    usingCombinedFilter_ = false;

    // Reset overlap buffers on rate change
    resetStreaming();

    std::cout << "Switched to rate family: " << rateFamilyToString(targetFamily) << std::endl;
    return true;
}

bool HRTFProcessor::initializeStreaming() {
    if (!initialized_) {
        return false;
    }

    resetStreaming();
    streamInitialized_ = true;
    return true;
}

void HRTFProcessor::resetStreaming() {
    std::fill(overlapBufferL_.begin(), overlapBufferL_.end(), 0.0f);
    std::fill(overlapBufferR_.begin(), overlapBufferR_.end(), 0.0f);

    if (d_overlapL_) {
        cudaMemset(d_overlapL_, 0, overlapSize_ * sizeof(float));
    }
    if (d_overlapR_) {
        cudaMemset(d_overlapR_, 0, overlapSize_ * sizeof(float));
    }
}

bool HRTFProcessor::processStereo(const float* inputL, const float* inputR,
                                   size_t inputFrames,
                                   std::vector<float>& outputL,
                                   std::vector<float>& outputR) {
    if (!initialized_ || !enabled_) {
        // Passthrough when disabled
        outputL.assign(inputL, inputL + inputFrames);
        outputR.assign(inputR, inputR + inputFrames);
        return true;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        outputL.resize(inputFrames);
        outputR.resize(inputFrames);

        // Process in blocks using Overlap-Save
        size_t outputPos = 0;
        size_t inputPos = 0;
        int threadsPerBlock = 256;

        while (outputPos < inputFrames) {
            size_t remainingSamples = inputFrames - inputPos;
            size_t currentBlockSize = std::min(remainingSamples,
                                               static_cast<size_t>(validOutputPerBlock_));

            // Prepare padded input: [overlap | new data]
            checkCudaError(cudaMemsetAsync(d_paddedInputL_, 0, fftSize_ * sizeof(float), stream_),
                           "memset padded L");
            checkCudaError(cudaMemsetAsync(d_paddedInputR_, 0, fftSize_ * sizeof(float), stream_),
                           "memset padded R");

            // Copy overlap
            if (overlapSize_ > 0 && outputPos > 0) {
                checkCudaError(
                    cudaMemcpyAsync(d_paddedInputL_, overlapBufferL_.data(),
                                    overlapSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
                    "copy overlap L");
                checkCudaError(
                    cudaMemcpyAsync(d_paddedInputR_, overlapBufferR_.data(),
                                    overlapSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
                    "copy overlap R");
            }

            // Copy new input data
            checkCudaError(
                cudaMemcpyAsync(d_paddedInputL_ + overlapSize_, inputL + inputPos,
                                currentBlockSize * sizeof(float), cudaMemcpyHostToDevice, stream_),
                "copy input L");
            checkCudaError(
                cudaMemcpyAsync(d_paddedInputR_ + overlapSize_, inputR + inputPos,
                                currentBlockSize * sizeof(float), cudaMemcpyHostToDevice, stream_),
                "copy input R");

            // Forward FFT of both inputs
            checkCufftError(cufftSetStream(fftPlanForward_, stream_), "set stream forward");
            checkCufftError(cufftExecR2C(fftPlanForward_, d_paddedInputL_, d_inputFFT_L_),
                            "FFT input L");
            checkCufftError(cufftExecR2C(fftPlanForward_, d_paddedInputR_, d_inputFFT_R_),
                            "FFT input R");

            // Convolutions in frequency domain
            // Out_L = In_L * LL + In_R * RL
            // Out_R = In_L * LR + In_R * RR
            int blocks = (filterFftSize_ + threadsPerBlock - 1) / threadsPerBlock;

            // d_convLL_ = L * LL
            complexMultiplyStoreKernel<<<blocks, threadsPerBlock, 0, stream_>>>(
                d_convLL_, d_inputFFT_L_, d_activeFilterFFT_[0], filterFftSize_);
            // d_convLL_ += R * RL -> d_convLL_ now contains Out_L
            complexMultiplyAccumulateKernel<<<blocks, threadsPerBlock, 0, stream_>>>(
                d_convLL_, d_inputFFT_R_, d_activeFilterFFT_[2], filterFftSize_);

            // d_convLR_ = L * LR
            complexMultiplyStoreKernel<<<blocks, threadsPerBlock, 0, stream_>>>(
                d_convLR_, d_inputFFT_L_, d_activeFilterFFT_[1], filterFftSize_);
            // d_convLR_ += R * RR -> d_convLR_ now contains Out_R
            complexMultiplyAccumulateKernel<<<blocks, threadsPerBlock, 0, stream_>>>(
                d_convLR_, d_inputFFT_R_, d_activeFilterFFT_[3], filterFftSize_);

            // Inverse FFT
            checkCufftError(cufftSetStream(fftPlanInverse_, stream_), "set stream inverse");
            checkCufftError(cufftExecC2R(fftPlanInverse_, d_convLL_, d_outputL_), "IFFT output L");
            checkCufftError(cufftExecC2R(fftPlanInverse_, d_convLR_, d_outputR_), "IFFT output R");

            // Scale
            float scale = 1.0f / fftSize_;
            int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
            scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream_>>>(d_outputL_, fftSize_, scale);
            scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream_>>>(d_outputR_, fftSize_, scale);

            // Extract valid output (skip first overlapSize_ samples)
            size_t validOutputSize = std::min(inputFrames - outputPos,
                                              static_cast<size_t>(validOutputPerBlock_));

            checkCudaError(
                cudaMemcpyAsync(outputL.data() + outputPos, d_outputL_ + overlapSize_,
                                validOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                "copy output L");
            checkCudaError(
                cudaMemcpyAsync(outputR.data() + outputPos, d_outputR_ + overlapSize_,
                                validOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                "copy output R");

            // Save overlap for next block using the actual block size that was processed
            if (overlapSize_ > 0) {
                size_t overlapCopyStart = currentBlockSize;
                size_t overlapCopyEnd = overlapCopyStart + overlapSize_;
                if (overlapCopyEnd <= static_cast<size_t>(fftSize_)) {
                    checkCudaError(
                        cudaMemcpyAsync(overlapBufferL_.data(), d_paddedInputL_ + overlapCopyStart,
                                        overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                        "save overlap L");
                    checkCudaError(
                        cudaMemcpyAsync(overlapBufferR_.data(), d_paddedInputR_ + overlapCopyStart,
                                        overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                        "save overlap R");
                }
            }

            outputPos += validOutputSize;
            inputPos += validOutputSize;
        }

        checkCudaError(cudaStreamSynchronize(stream_), "stream sync");

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        stats_.totalProcessingTime += elapsed.count();
        stats_.framesProcessed += inputFrames;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in processStereo: " << e.what() << std::endl;
        return false;
    }
}

bool HRTFProcessor::processStreamBlock(const float* inputL, const float* inputR,
                                         size_t inputFrames,
                                         std::vector<float>& outputL,
                                         std::vector<float>& outputR,
                                         cudaStream_t stream,
                                         std::vector<float>& streamInputBufferL,
                                         std::vector<float>& streamInputBufferR,
                                         size_t& streamInputAccumulatedL,
                                         size_t& streamInputAccumulatedR) {
    if (!initialized_ || !streamInitialized_) {
        return false;
    }

    if (!enabled_) {
        // Passthrough
        outputL.assign(inputL, inputL + inputFrames);
        outputR.assign(inputR, inputR + inputFrames);
        return true;
    }

    // Detect if output buffers are different vector instances from previous call
    // (e.g., new local variables passed each time). Clear stale tracked pointers
    // to prevent cudaHostUnregister on memory that no longer belongs to us.
    if (pinnedStreamOutputL_ != nullptr &&
        (outputL.empty() || pinnedStreamOutputL_ != outputL.data())) {
        safeCudaHostUnregister(&pinnedStreamOutputL_, &pinnedStreamOutputLBytes_,
                               "cudaHostUnregister stale crossfeed stream output L");
    }
    if (pinnedStreamOutputR_ != nullptr &&
        (outputR.empty() || pinnedStreamOutputR_ != outputR.data())) {
        safeCudaHostUnregister(&pinnedStreamOutputR_, &pinnedStreamOutputRBytes_,
                               "cudaHostUnregister stale crossfeed stream output R");
    }

    // Accumulate input
    // Before resize, save old pointers to detect reallocation
    if (streamInputBufferL.size() < streamInputAccumulatedL + inputFrames) {
        void* oldPtrL = streamInputBufferL.data();
        void* oldPtrR = streamInputBufferR.data();

        streamInputBufferL.resize(streamInputAccumulatedL + inputFrames + streamValidInputPerBlock_);
        streamInputBufferR.resize(streamInputAccumulatedR + inputFrames + streamValidInputPerBlock_);

        // If pointers changed, old memory was freed by vector
        // Clear tracked pointers to prevent cudaHostUnregister on freed memory
        if (streamInputBufferL.data() != oldPtrL) {
            pinnedStreamInputL_ = nullptr;
            pinnedStreamInputLBytes_ = 0;
        }
        if (streamInputBufferR.data() != oldPtrR) {
            pinnedStreamInputR_ = nullptr;
            pinnedStreamInputRBytes_ = 0;
        }
    }
    registerStreamBuffer(streamInputBufferL, &pinnedStreamInputL_, &pinnedStreamInputLBytes_,
                         "cudaHostRegister crossfeed stream input L");
    registerStreamBuffer(streamInputBufferR, &pinnedStreamInputR_, &pinnedStreamInputRBytes_,
                         "cudaHostRegister crossfeed stream input R");

    std::memcpy(streamInputBufferL.data() + streamInputAccumulatedL,
                inputL, inputFrames * sizeof(float));
    std::memcpy(streamInputBufferR.data() + streamInputAccumulatedR,
                inputR, inputFrames * sizeof(float));
    streamInputAccumulatedL += inputFrames;
    streamInputAccumulatedR += inputFrames;

    // Check if we have enough for a block
    if (streamInputAccumulatedL < streamValidInputPerBlock_) {
        outputL.clear();
        outputR.clear();
        return false;  // Not enough data yet
    }

    // Process one block
    // Protect output buffers from pointer invalidation on resize
    {
        void* oldPtrL = outputL.data();
        void* oldPtrR = outputR.data();

        outputL.resize(validOutputPerBlock_);
        outputR.resize(validOutputPerBlock_);

        // If pointers changed, old memory was freed by vector
        // Clear tracked pointers to prevent cudaHostUnregister on freed memory
        if (outputL.data() != oldPtrL) {
            pinnedStreamOutputL_ = nullptr;
            pinnedStreamOutputLBytes_ = 0;
        }
        if (outputR.data() != oldPtrR) {
            pinnedStreamOutputR_ = nullptr;
            pinnedStreamOutputRBytes_ = 0;
        }
    }
    registerStreamBuffer(outputL, &pinnedStreamOutputL_, &pinnedStreamOutputLBytes_,
                         "cudaHostRegister crossfeed stream output L");
    registerStreamBuffer(outputR, &pinnedStreamOutputR_, &pinnedStreamOutputRBytes_,
                         "cudaHostRegister crossfeed stream output R");

    int threadsPerBlock = 256;
    int blocks = (filterFftSize_ + threadsPerBlock - 1) / threadsPerBlock;

    // Prepare padded input with device-resident overlap
    checkCudaError(cudaMemsetAsync(d_paddedInputL_, 0, fftSize_ * sizeof(float), stream),
                   "memset padded L");
    checkCudaError(cudaMemsetAsync(d_paddedInputR_, 0, fftSize_ * sizeof(float), stream),
                   "memset padded R");

    // Copy device overlap
    checkCudaError(
        cudaMemcpyAsync(d_paddedInputL_, d_overlapL_, overlapSize_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream),
        "copy device overlap L");
    checkCudaError(
        cudaMemcpyAsync(d_paddedInputR_, d_overlapR_, overlapSize_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream),
        "copy device overlap R");

    // Copy new input
    checkCudaError(
        cudaMemcpyAsync(d_paddedInputL_ + overlapSize_, streamInputBufferL.data(),
                        streamValidInputPerBlock_ * sizeof(float), cudaMemcpyHostToDevice, stream),
        "copy stream input L");
    checkCudaError(
        cudaMemcpyAsync(d_paddedInputR_ + overlapSize_, streamInputBufferR.data(),
                        streamValidInputPerBlock_ * sizeof(float), cudaMemcpyHostToDevice, stream),
        "copy stream input R");

    // Forward FFT
    checkCufftError(cufftSetStream(fftPlanForward_, stream), "set stream forward");
    checkCufftError(cufftExecR2C(fftPlanForward_, d_paddedInputL_, d_inputFFT_L_), "FFT L");
    checkCufftError(cufftExecR2C(fftPlanForward_, d_paddedInputR_, d_inputFFT_R_), "FFT R");

    // 4-channel convolution:
    // Out_L = L*LL + R*RL
    // Out_R = L*LR + R*RR

    // d_convLL_ = L * LL
    complexMultiplyStoreKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_convLL_, d_inputFFT_L_, d_activeFilterFFT_[0], filterFftSize_);
    // d_convLL_ += R * RL
    complexMultiplyAccumulateKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_convLL_, d_inputFFT_R_, d_activeFilterFFT_[2], filterFftSize_);

    // d_convLR_ = L * LR
    complexMultiplyStoreKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_convLR_, d_inputFFT_L_, d_activeFilterFFT_[1], filterFftSize_);
    // d_convLR_ += R * RR
    complexMultiplyAccumulateKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_convLR_, d_inputFFT_R_, d_activeFilterFFT_[3], filterFftSize_);

    // Inverse FFT
    checkCufftError(cufftSetStream(fftPlanInverse_, stream), "set stream inverse");
    checkCufftError(cufftExecC2R(fftPlanInverse_, d_convLL_, d_outputL_), "IFFT L");
    checkCufftError(cufftExecC2R(fftPlanInverse_, d_convLR_, d_outputR_), "IFFT R");

    // Scale
    float scale = 1.0f / fftSize_;
    int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(d_outputL_, fftSize_, scale);
    scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(d_outputR_, fftSize_, scale);

    // Copy valid output (asynchronous, wait before touching host buffers again)
    checkCudaError(
        cudaMemcpyAsync(outputL.data(), d_outputL_ + overlapSize_,
                        validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "copy output L");
    checkCudaError(
        cudaMemcpyAsync(outputR.data(), d_outputR_ + overlapSize_,
                        validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "copy output R");

    // Update device overlap using the actual tail of this block
    if (overlapSize_ > 0) {
        size_t overlapSamples =
            std::min(static_cast<size_t>(overlapSize_), static_cast<size_t>(streamValidInputPerBlock_));
        size_t overlapStart = streamValidInputPerBlock_ - overlapSamples;
        checkCudaError(
            cudaMemcpyAsync(d_overlapL_, d_paddedInputL_ + overlapSize_ + overlapStart,
                            overlapSamples * sizeof(float), cudaMemcpyDeviceToDevice, stream),
            "update overlap L");
        checkCudaError(
            cudaMemcpyAsync(d_overlapR_, d_paddedInputR_ + overlapSize_ + overlapStart,
                            overlapSamples * sizeof(float), cudaMemcpyDeviceToDevice, stream),
            "update overlap R");
    }

    // Ensure all device↔host transfers finished before reusing/altering host buffers
    checkCudaError(cudaStreamSynchronize(stream), "stream sync (crossfeed streaming)");

    // Shift input buffer
    size_t remaining = streamInputAccumulatedL - streamValidInputPerBlock_;
    if (remaining > 0) {
        std::memmove(streamInputBufferL.data(),
                     streamInputBufferL.data() + streamValidInputPerBlock_,
                     remaining * sizeof(float));
        std::memmove(streamInputBufferR.data(),
                     streamInputBufferR.data() + streamValidInputPerBlock_,
                     remaining * sizeof(float));
    }
    streamInputAccumulatedL = remaining;
    streamInputAccumulatedR = remaining;

    return true;
}

const HRTFMetadata& HRTFProcessor::getCurrentMetadata() const {
    static HRTFMetadata empty;
    if (!initialized_) {
        return empty;
    }
    return metadata_[activeFilterConfig_];
}

void HRTFProcessor::cleanup() {
    // Free filter FFT buffers
    for (int i = 0; i < NUM_CONFIGS; ++i) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            if (d_filterFFT_[i][c]) {
                cudaFree(d_filterFFT_[i][c]);
                d_filterFFT_[i][c] = nullptr;
            }
        }
    }

    // Free combined filter FFT buffers
    for (int f = 0; f < NUM_RATE_FAMILIES; ++f) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            if (d_combinedFilterFFT_[f][c]) {
                cudaFree(d_combinedFilterFFT_[f][c]);
                d_combinedFilterFFT_[f][c] = nullptr;
            }
        }
        combinedFilterLoaded_[f] = false;
    }
    usingCombinedFilter_ = false;

    // Free working buffers
    if (d_inputL_) cudaFree(d_inputL_);
    if (d_inputR_) cudaFree(d_inputR_);
    if (d_paddedInputL_) cudaFree(d_paddedInputL_);
    if (d_paddedInputR_) cudaFree(d_paddedInputR_);
    if (d_inputFFT_L_) cudaFree(d_inputFFT_L_);
    if (d_inputFFT_R_) cudaFree(d_inputFFT_R_);
    if (d_convLL_) cudaFree(d_convLL_);
    if (d_convLR_) cudaFree(d_convLR_);
    if (d_convRL_) cudaFree(d_convRL_);
    if (d_convRR_) cudaFree(d_convRR_);
    if (d_outputL_) cudaFree(d_outputL_);
    if (d_outputR_) cudaFree(d_outputR_);
    if (d_tempConv_) cudaFree(d_tempConv_);
    if (d_overlapL_) cudaFree(d_overlapL_);
    if (d_overlapR_) cudaFree(d_overlapR_);

    // Destroy cuFFT plans
    if (fftPlanForward_) cufftDestroy(fftPlanForward_);
    if (fftPlanInverse_) cufftDestroy(fftPlanInverse_);

    // Destroy streams
    if (stream_) cudaStreamDestroy(stream_);
    if (streamL_) cudaStreamDestroy(streamL_);
    if (streamR_) cudaStreamDestroy(streamR_);

    // Unregister pinned host buffers
    safeCudaHostUnregister(&pinnedStreamInputL_, &pinnedStreamInputLBytes_,
                           "cudaHostUnregister cleanup input L");
    safeCudaHostUnregister(&pinnedStreamInputR_, &pinnedStreamInputRBytes_,
                           "cudaHostUnregister cleanup input R");
    safeCudaHostUnregister(&pinnedStreamOutputL_, &pinnedStreamOutputLBytes_,
                           "cudaHostUnregister cleanup output L");
    safeCudaHostUnregister(&pinnedStreamOutputR_, &pinnedStreamOutputRBytes_,
                           "cudaHostUnregister cleanup output R");

    // Reset pointers
    d_inputL_ = nullptr;
    d_inputR_ = nullptr;
    d_paddedInputL_ = nullptr;
    d_paddedInputR_ = nullptr;
    d_inputFFT_L_ = nullptr;
    d_inputFFT_R_ = nullptr;
    d_convLL_ = nullptr;
    d_convLR_ = nullptr;
    d_convRL_ = nullptr;
    d_convRR_ = nullptr;
    d_outputL_ = nullptr;
    d_outputR_ = nullptr;
    d_tempConv_ = nullptr;
    d_overlapL_ = nullptr;
    d_overlapR_ = nullptr;
    fftPlanForward_ = 0;
    fftPlanInverse_ = 0;
    stream_ = nullptr;
    streamL_ = nullptr;
    streamR_ = nullptr;

    for (int c = 0; c < NUM_CHANNELS; ++c) {
        d_activeFilterFFT_[c] = nullptr;
    }
    activeFilterConfig_ = -1;

    // Clear host data
    for (int i = 0; i < NUM_CONFIGS; ++i) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            h_filterCoeffs_[i][c].clear();
        }
    }
    overlapBufferL_.clear();
    overlapBufferR_.clear();

    initialized_ = false;
    streamInitialized_ = false;
}

bool HRTFProcessor::setCombinedFilter(RateFamily rateFamily,
                                       const cufftComplex* combinedLL,
                                       const cufftComplex* combinedLR,
                                       const cufftComplex* combinedRL,
                                       const cufftComplex* combinedRR,
                                       size_t filterComplexCount) {
    if (!initialized_) {
        std::cerr << "HRTFProcessor::setCombinedFilter: Not initialized" << std::endl;
        return false;
    }

    if (rateFamily == RateFamily::RATE_UNKNOWN) {
        std::cerr << "HRTFProcessor::setCombinedFilter: Invalid rate family" << std::endl;
        return false;
    }

    // Validate filter size matches expected FFT size
    if (filterComplexCount != filterFftSize_) {
        std::cerr << "HRTFProcessor::setCombinedFilter: Filter size mismatch. "
                  << "Expected " << filterFftSize_ << " complex values, got " << filterComplexCount
                  << std::endl;
        return false;
    }

    int familyIdx = static_cast<int>(rateFamily);
    size_t filterBytes = filterComplexCount * sizeof(cufftComplex);

    // Allocate device memory if not already allocated
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        if (d_combinedFilterFFT_[familyIdx][c] == nullptr) {
            cudaError_t err = cudaMalloc(&d_combinedFilterFFT_[familyIdx][c], filterBytes);
            if (err != cudaSuccess) {
                std::cerr << "HRTFProcessor::setCombinedFilter: cudaMalloc failed for channel " << c
                          << ": " << cudaGetErrorString(err) << std::endl;
                // Clean up any partially allocated memory
                for (int cc = 0; cc < c; ++cc) {
                    if (d_combinedFilterFFT_[familyIdx][cc]) {
                        cudaFree(d_combinedFilterFFT_[familyIdx][cc]);
                        d_combinedFilterFFT_[familyIdx][cc] = nullptr;
                    }
                }
                return false;
            }
        }
    }

    // Copy filter data from host to device
    const cufftComplex* srcFilters[NUM_CHANNELS] = {combinedLL, combinedLR, combinedRL, combinedRR};
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        cudaError_t err = cudaMemcpy(d_combinedFilterFFT_[familyIdx][c], srcFilters[c], filterBytes,
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "HRTFProcessor::setCombinedFilter: cudaMemcpy failed for channel " << c
                      << ": " << cudaGetErrorString(err) << std::endl;
            // Rollback: invalidate this combined filter to prevent use of corrupted data
            combinedFilterLoaded_[familyIdx] = false;
            // If this was the active filter, revert to predefined
            if (rateFamily == currentRateFamily_ && usingCombinedFilter_) {
                int config = getFilterIndex(currentHeadSize_, currentRateFamily_);
                if (d_filterFFT_[config][0] != nullptr) {
                    for (int cc = 0; cc < NUM_CHANNELS; ++cc) {
                        d_activeFilterFFT_[cc] = d_filterFFT_[config][cc];
                    }
                }
                usingCombinedFilter_ = false;
                resetStreaming();
                std::cerr << "HRTFProcessor::setCombinedFilter: Reverted to predefined filter"
                          << std::endl;
            }
            return false;
        }
    }

    combinedFilterLoaded_[familyIdx] = true;

    // If this is for the current rate family, switch to combined filter
    if (rateFamily == currentRateFamily_) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            d_activeFilterFFT_[c] = d_combinedFilterFFT_[familyIdx][c];
        }
        usingCombinedFilter_ = true;
        // Clear overlap buffers to prevent old filter tail from mixing in
        resetStreaming();
    }

    std::cout << "HRTFProcessor: Set combined filter for " << rateFamilyToString(rateFamily)
              << " (" << filterComplexCount << " complex values)" << std::endl;
    return true;
}

bool HRTFProcessor::generateWoodworthProfile(RateFamily rateFamily, float azimuthDeg,
                                             const HRTF::WoodworthParams& params) {
    if (!initialized_) {
        std::cerr << "HRTFProcessor::generateWoodworthProfile: Not initialized" << std::endl;
        return false;
    }
    if (rateFamily == RateFamily::RATE_UNKNOWN) {
        std::cerr << "HRTFProcessor::generateWoodworthProfile: Invalid rate family" << std::endl;
        return false;
    }
    if (filterTaps_ <= 0 || fftSize_ <= 0) {
        std::cerr << "HRTFProcessor::generateWoodworthProfile: Invalid filter geometry" << std::endl;
        return false;
    }

    HRTF::WoodworthParams tuned = params;
    tuned.taps = static_cast<size_t>(filterTaps_);
    tuned.sampleRate = (rateFamily == RateFamily::RATE_44K) ? 44100.0f : 48000.0f;

    HRTF::WoodworthIRSet irSet = HRTF::generateWoodworthSet(azimuthDeg, tuned);

    auto normalize = [&](const std::vector<float>& src) {
        std::vector<float> dst(filterTaps_, 0.0f);
        size_t copyCount = std::min(dst.size(), src.size());
        if (copyCount > 0) {
            std::copy(src.begin(), src.begin() + copyCount, dst.begin());
        }
        return dst;
    };

    std::vector<float> tdLL = normalize(irSet.ll);
    std::vector<float> tdLR = normalize(irSet.lr);
    std::vector<float> tdRL = normalize(irSet.rl);
    std::vector<float> tdRR = normalize(irSet.rr);

    float* d_temp = nullptr;
    cufftComplex* d_fftTemp = nullptr;
    std::vector<cufftComplex> freqLL;
    std::vector<cufftComplex> freqLR;
    std::vector<cufftComplex> freqRL;
    std::vector<cufftComplex> freqRR;

    auto cleanup = [&]() {
        if (d_temp) {
            cudaFree(d_temp);
            d_temp = nullptr;
        }
        if (d_fftTemp) {
            cudaFree(d_fftTemp);
            d_fftTemp = nullptr;
        }
    };

    try {
        checkCudaError(cudaMalloc(&d_temp, fftSize_ * sizeof(float)),
                       "cudaMalloc woodworth temp");
        checkCudaError(cudaMalloc(&d_fftTemp, filterFftSize_ * sizeof(cufftComplex)),
                       "cudaMalloc woodworth FFT");

        auto convertChannel = [&](const std::vector<float>& timeDomain,
                                  std::vector<cufftComplex>& freqDomain,
                                  const char* context) {
            cudaStream_t workStream = stream_ ? stream_ : nullptr;
            size_t copyBytes =
                std::min(static_cast<size_t>(filterTaps_), timeDomain.size()) * sizeof(float);
            checkCudaError(
                cudaMemsetAsync(d_temp, 0, fftSize_ * sizeof(float), workStream),
                "cudaMemsetAsync woodworth temp");
            if (copyBytes > 0) {
                checkCudaError(
                    cudaMemcpyAsync(d_temp, timeDomain.data(), copyBytes, cudaMemcpyHostToDevice,
                                    workStream),
                    "cudaMemcpyAsync woodworth temp");
            }
            checkCufftError(cufftSetStream(fftPlanForward_, workStream), "cufftSetStream woodworth");
            checkCufftError(cufftExecR2C(fftPlanForward_, d_temp, d_fftTemp), context);
            checkCudaError(cudaStreamSynchronize(workStream), "cudaStreamSynchronize woodworth");
            freqDomain.resize(filterFftSize_);
            checkCudaError(
                cudaMemcpy(freqDomain.data(), d_fftTemp,
                           filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
                "cudaMemcpy woodworth FFT");
        };

        convertChannel(tdLL, freqLL, "cufftExecR2C woodworth LL");
        convertChannel(tdLR, freqLR, "cufftExecR2C woodworth LR");
        convertChannel(tdRL, freqRL, "cufftExecR2C woodworth RL");
        convertChannel(tdRR, freqRR, "cufftExecR2C woodworth RR");

        cleanup();

        bool success = setCombinedFilter(rateFamily, freqLL.data(), freqLR.data(), freqRL.data(),
                                         freqRR.data(), filterFftSize_);
        if (success) {
            std::cout << "HRTFProcessor: Generated Woodworth profile (azimuth=" << azimuthDeg
                      << " deg, family=" << rateFamilyToString(rateFamily) << ")" << std::endl;
        }
        return success;

    } catch (const std::exception& e) {
        std::cerr << "HRTFProcessor::generateWoodworthProfile failed: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void HRTFProcessor::clearCombinedFilter() {
    if (!initialized_) {
        return;
    }

    // If currently using combined filter, switch back to predefined
    if (usingCombinedFilter_) {
        int config = getFilterIndex(currentHeadSize_, currentRateFamily_);
        if (d_filterFFT_[config][0] != nullptr) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                d_activeFilterFFT_[c] = d_filterFFT_[config][c];
            }
        }
        usingCombinedFilter_ = false;
        // Clear overlap buffers to prevent old filter tail from mixing in
        resetStreaming();
        std::cout << "HRTFProcessor: Cleared combined filter, reverted to predefined HRTF"
                  << std::endl;
    }

    // Free combined filter memory
    for (int f = 0; f < NUM_RATE_FAMILIES; ++f) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            if (d_combinedFilterFFT_[f][c]) {
                cudaFree(d_combinedFilterFFT_[f][c]);
                d_combinedFilterFFT_[f][c] = nullptr;
            }
        }
        combinedFilterLoaded_[f] = false;
    }
}

void HRTFProcessor::releaseHostCoefficients() {
    // Release all CPU-side HRTF coefficient vectors to free memory
    // This is called after GPU FFT transfer is complete
    // Important for Jetson Unified Memory optimization

    size_t freedBytes = 0;

    for (int i = 0; i < NUM_CONFIGS; ++i) {
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            if (!h_filterCoeffs_[i][c].empty()) {
                freedBytes += h_filterCoeffs_[i][c].capacity() * sizeof(float);
                h_filterCoeffs_[i][c].clear();
                h_filterCoeffs_[i][c].shrink_to_fit();
            }
        }
    }

    if (freedBytes > 0) {
        std::cout << "  Released HRTF CPU coefficient memory: "
                  << (freedBytes / 1024) << " KB ("
                  << freedBytes << " bytes)" << std::endl;
    }
}

}  // namespace CrossfeedEngine
