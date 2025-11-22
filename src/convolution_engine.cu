#include "convolution_engine.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <memory>
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

namespace {

// RAII for short-lived host buffers; long-lived buffers are registered via
// registerHostBuffer/unregisterHostBuffers to avoid repetitive cudaHostRegister calls.
class ScopedHostPin {
public:
    ScopedHostPin(void* ptr, size_t bytes, const char* context)
        : ptr_(ptr), registered_(false) {
        if (ptr_ && bytes > 0) {
            ConvolutionEngine::Utils::checkCudaError(
                cudaHostRegister(ptr_, bytes, cudaHostRegisterDefault),
                context
            );
            registered_ = true;
        }
    }

    ~ScopedHostPin() {
        if (registered_) {
            cudaHostUnregister(ptr_);
        }
    }

    ScopedHostPin(const ScopedHostPin&) = delete;
    ScopedHostPin& operator=(const ScopedHostPin&) = delete;

private:
    void* ptr_;
    bool registered_;
};

} // namespace

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
      pinnedStreamInputMonoBytes_(0) {
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

bool GPUUpsampler::initializeDualRate(const std::string& filterCoeffPath44k,
                                       const std::string& filterCoeffPath48k,
                                       int upsampleRatio,
                                       int blockSize,
                                       RateFamily initialFamily) {
    upsampleRatio_ = upsampleRatio;
    blockSize_ = blockSize;
    currentRateFamily_ = initialFamily;

    std::cout << "Initializing GPU Upsampler (Dual-Rate Mode)..." << std::endl;
    std::cout << "  Upsample Ratio: " << upsampleRatio_ << "x" << std::endl;
    std::cout << "  Block Size: " << blockSize_ << " samples" << std::endl;
    std::cout << "  Initial Rate Family: "
              << (initialFamily == RateFamily::RATE_44K ? "44.1kHz" : "48kHz") << std::endl;

    // Load 44.1kHz family coefficients
    std::cout << "Loading 44.1kHz family coefficients..." << std::endl;
    {
        std::ifstream ifs(filterCoeffPath44k, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error: Cannot open 44.1kHz coefficients: " << filterCoeffPath44k << std::endl;
            return false;
        }
        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        int taps44k = fileSize / sizeof(float);
        h_filterCoeffs44k_.resize(taps44k);
        ifs.read(reinterpret_cast<char*>(h_filterCoeffs44k_.data()), fileSize);
        std::cout << "  44.1kHz: " << taps44k << " taps (" << fileSize / 1024 << " KB)" << std::endl;
    }

    // Load 48kHz family coefficients
    std::cout << "Loading 48kHz family coefficients..." << std::endl;
    {
        std::ifstream ifs(filterCoeffPath48k, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error: Cannot open 48kHz coefficients: " << filterCoeffPath48k << std::endl;
            return false;
        }
        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        int taps48k = fileSize / sizeof(float);
        h_filterCoeffs48k_.resize(taps48k);
        ifs.read(reinterpret_cast<char*>(h_filterCoeffs48k_.data()), fileSize);
        std::cout << "  48kHz: " << taps48k << " taps (" << fileSize / 1024 << " KB)" << std::endl;
    }

    // Verify both coefficient sets have the same tap count
    if (h_filterCoeffs44k_.size() != h_filterCoeffs48k_.size()) {
        std::cerr << "Error: Coefficient tap counts do not match (44k: "
                  << h_filterCoeffs44k_.size() << ", 48k: " << h_filterCoeffs48k_.size() << ")" << std::endl;
        return false;
    }

    // Use 44k coefficients as primary for initialization
    filterTaps_ = h_filterCoeffs44k_.size();
    h_filterCoeffs_ = (initialFamily == RateFamily::RATE_44K) ? h_filterCoeffs44k_ : h_filterCoeffs48k_;

    // Setup GPU resources (this will pre-compute FFT for current family)
    if (!setupGPUResources()) {
        return false;
    }

    // Pre-compute FFT for both coefficient sets
    std::cout << "Pre-computing FFT for both rate families..." << std::endl;

    // Allocate GPU buffers for both families
    Utils::checkCudaError(
        cudaMalloc(&d_filterFFT_44k_, filterFftSize_ * sizeof(cufftComplex)),
        "cudaMalloc d_filterFFT_44k_"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_filterFFT_48k_, filterFftSize_ * sizeof(cufftComplex)),
        "cudaMalloc d_filterFFT_48k_"
    );

    // Compute FFT for 44.1kHz coefficients
    {
        float* d_temp;
        Utils::checkCudaError(
            cudaMalloc(&d_temp, fftSize_ * sizeof(float)),
            "cudaMalloc temp for 44k FFT"
        );
        Utils::checkCudaError(
            cudaMemset(d_temp, 0, fftSize_ * sizeof(float)),
            "cudaMemset temp"
        );
        Utils::checkCudaError(
            cudaMemcpy(d_temp, h_filterCoeffs44k_.data(), filterTaps_ * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy 44k coeffs to device"
        );
        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_44k_),
            "cufftExecR2C for 44k"
        );
        cudaFree(d_temp);
    }

    // Compute FFT for 48kHz coefficients
    {
        float* d_temp;
        Utils::checkCudaError(
            cudaMalloc(&d_temp, fftSize_ * sizeof(float)),
            "cudaMalloc temp for 48k FFT"
        );
        Utils::checkCudaError(
            cudaMemset(d_temp, 0, fftSize_ * sizeof(float)),
            "cudaMemset temp"
        );
        Utils::checkCudaError(
            cudaMemcpy(d_temp, h_filterCoeffs48k_.data(), filterTaps_ * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy 48k coeffs to device"
        );
        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_48k_),
            "cufftExecR2C for 48k"
        );
        cudaFree(d_temp);
    }

    // Set the active filter FFT to the initial family
    d_activeFilterFFT_ = (initialFamily == RateFamily::RATE_44K) ? d_filterFFT_44k_ : d_filterFFT_48k_;

    // Copy to the original filter FFT for EQ restoration
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, d_activeFilterFFT_,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy to originalFilterFFT"
    );

    // Also copy to the A/B buffers for ping-pong
    Utils::checkCudaError(
        cudaMemcpy(d_filterFFT_A_, d_activeFilterFFT_,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy to filterFFT_A"
    );
    d_activeFilterFFT_ = d_filterFFT_A_;

    dualRateEnabled_ = true;
    std::cout << "GPU Upsampler (Dual-Rate) initialized successfully!" << std::endl;
    return true;
}

bool GPUUpsampler::switchRateFamily(RateFamily targetFamily) {
    if (!dualRateEnabled_) {
        std::cerr << "Error: Dual-rate mode not enabled" << std::endl;
        return false;
    }

    if (targetFamily == currentRateFamily_) {
        std::cout << "Already at target rate family" << std::endl;
        return true;
    }

    if (targetFamily == RateFamily::RATE_UNKNOWN) {
        std::cerr << "Error: Cannot switch to unknown rate family" << std::endl;
        return false;
    }

    // TODO: Implement Soft Mute (fade-out/fade-in) to prevent pop noise during switch
    // See: https://github.com/michihitoTakami/michy_os/issues/38
    // Current implementation uses double-buffering but no audio fade.

    std::cout << "Switching rate family: "
              << (currentRateFamily_ == RateFamily::RATE_44K ? "44.1kHz" : "48kHz")
              << " -> "
              << (targetFamily == RateFamily::RATE_44K ? "44.1kHz" : "48kHz")
              << std::endl;

    // Select the source FFT for the target family
    cufftComplex* sourceFFT = (targetFamily == RateFamily::RATE_44K) ? d_filterFFT_44k_ : d_filterFFT_48k_;

    // Use double-buffering for glitch-free switching
    // Copy to the inactive buffer, then swap
    cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_) ? d_filterFFT_B_ : d_filterFFT_A_;

    Utils::checkCudaError(
        cudaMemcpyAsync(backBuffer, sourceFFT,
                        filterFftSize_ * sizeof(cufftComplex),
                        cudaMemcpyDeviceToDevice, stream_),
        "cudaMemcpyAsync rate family switch"
    );

    // Synchronize to ensure copy is complete before switching
    Utils::checkCudaError(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize rate family switch"
    );

    // Atomic swap to the new buffer
    d_activeFilterFFT_ = backBuffer;

    // Update original filter FFT for EQ restoration
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, sourceFFT,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy update originalFilterFFT"
    );

    // Update host coefficients reference
    h_filterCoeffs_ = (targetFamily == RateFamily::RATE_44K) ? h_filterCoeffs44k_ : h_filterCoeffs48k_;

    // Clear EQ state (EQ needs to be re-applied for new rate family)
    if (eqApplied_) {
        std::cout << "  Note: EQ was applied. It needs to be re-applied for the new rate family." << std::endl;
        eqApplied_ = false;
    }

    currentRateFamily_ = targetFamily;
    std::cout << "Rate family switch complete" << std::endl;
    return true;
}

bool GPUUpsampler::initializeMultiRate(const std::string& coefficientDir,
                                        int blockSize,
                                        int initialInputRate) {
    blockSize_ = blockSize;
    currentInputRate_ = initialInputRate;

    // Find the config index for the initial input rate
    currentMultiRateIndex_ = findMultiRateConfigIndex(initialInputRate);
    if (currentMultiRateIndex_ < 0) {
        std::cerr << "Error: Unsupported initial input rate: " << initialInputRate << std::endl;
        return false;
    }

    const auto& initialConfig = MULTI_RATE_CONFIGS[currentMultiRateIndex_];
    upsampleRatio_ = initialConfig.ratio;
    currentRateFamily_ = initialConfig.family;

    std::cout << "Initializing GPU Upsampler (Multi-Rate Mode)..." << std::endl;
    std::cout << "  Coefficient Directory: " << coefficientDir << std::endl;
    std::cout << "  Block Size: " << blockSize_ << " samples" << std::endl;
    std::cout << "  Initial Input Rate: " << initialInputRate << " Hz" << std::endl;
    std::cout << "  Initial Upsample Ratio: " << upsampleRatio_ << "x" << std::endl;

    // Load all 8 filter coefficient files
    int loadedCount = 0;
    int maxTaps = 0;

    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        const auto& config = MULTI_RATE_CONFIGS[i];

        // Construct filename: e.g., filter_44k_16x_1024_min_phase.bin
        std::string familyStr = (config.family == RateFamily::RATE_44K) ? "44k" : "48k";
        std::string filename = coefficientDir + "/filter_" + familyStr + "_" +
                               std::to_string(config.ratio) + "x_";

        // Search for matching file (tap count may vary)
        std::string foundPath;
        for (int taps : {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2000000}) {
            std::string testPath = filename + std::to_string(taps) + "_min_phase.bin";
            std::ifstream testFile(testPath, std::ios::binary);
            if (testFile.good()) {
                foundPath = testPath;
                break;
            }
        }

        if (foundPath.empty()) {
            std::cerr << "Error: Cannot find coefficient file for " << familyStr
                      << " " << config.ratio << "x in " << coefficientDir << std::endl;
            return false;
        }

        // Load coefficients
        std::ifstream ifs(foundPath, std::ios::binary);
        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        int taps = fileSize / sizeof(float);
        h_filterCoeffsMulti_[i].resize(taps);
        ifs.read(reinterpret_cast<char*>(h_filterCoeffsMulti_[i].data()), fileSize);

        if (taps > maxTaps) {
            maxTaps = taps;
        }

        std::cout << "  Loaded " << familyStr << "_" << config.ratio << "x: "
                  << taps << " taps (" << fileSize / 1024 << " KB)" << std::endl;
        ++loadedCount;
    }

    if (loadedCount != MULTI_RATE_CONFIG_COUNT) {
        std::cerr << "Error: Failed to load all coefficient files" << std::endl;
        return false;
    }

    // Use the initial config's coefficients as primary
    filterTaps_ = h_filterCoeffsMulti_[currentMultiRateIndex_].size();
    h_filterCoeffs_ = h_filterCoeffsMulti_[currentMultiRateIndex_];

    // Setup GPU resources
    if (!setupGPUResources()) {
        return false;
    }

    // Pre-compute FFT for all 8 coefficient sets
    std::cout << "Pre-computing FFT for all " << MULTI_RATE_CONFIG_COUNT << " rate configurations..." << std::endl;

    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        const auto& config = MULTI_RATE_CONFIGS[i];
        const auto& coeffs = h_filterCoeffsMulti_[i];

        // Allocate GPU buffer for this config's FFT
        Utils::checkCudaError(
            cudaMalloc(&d_filterFFT_Multi_[i], filterFftSize_ * sizeof(cufftComplex)),
            "cudaMalloc d_filterFFT_Multi_"
        );

        // Compute FFT
        float* d_temp;
        Utils::checkCudaError(
            cudaMalloc(&d_temp, fftSize_ * sizeof(float)),
            "cudaMalloc temp for multi-rate FFT"
        );
        Utils::checkCudaError(
            cudaMemset(d_temp, 0, fftSize_ * sizeof(float)),
            "cudaMemset temp"
        );
        Utils::checkCudaError(
            cudaMemcpy(d_temp, coeffs.data(), coeffs.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy coeffs to device"
        );
        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_Multi_[i]),
            "cufftExecR2C for multi-rate"
        );
        cudaFree(d_temp);

        std::string familyStr = (config.family == RateFamily::RATE_44K) ? "44k" : "48k";
        std::cout << "  FFT computed for " << familyStr << "_" << config.ratio << "x" << std::endl;
    }

    // Set the active filter FFT to the initial configuration
    d_activeFilterFFT_ = d_filterFFT_Multi_[currentMultiRateIndex_];

    // Copy to the original filter FFT for EQ restoration
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, d_activeFilterFFT_,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy to originalFilterFFT"
    );

    // Also copy to the A buffer for ping-pong
    Utils::checkCudaError(
        cudaMemcpy(d_filterFFT_A_, d_activeFilterFFT_,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy to filterFFT_A"
    );
    d_activeFilterFFT_ = d_filterFFT_A_;

    multiRateEnabled_ = true;
    std::cout << "GPU Upsampler (Multi-Rate) initialized successfully!" << std::endl;
    return true;
}

bool GPUUpsampler::switchToInputRate(int inputSampleRate) {
    if (!multiRateEnabled_) {
        std::cerr << "Error: Multi-rate mode not enabled" << std::endl;
        return false;
    }

    int targetIndex = findMultiRateConfigIndex(inputSampleRate);
    if (targetIndex < 0) {
        std::cerr << "Error: Unsupported input sample rate: " << inputSampleRate << std::endl;
        return false;
    }

    if (targetIndex == currentMultiRateIndex_) {
        std::cout << "Already at target input rate: " << inputSampleRate << " Hz" << std::endl;
        return true;
    }

    const auto& currentConfig = MULTI_RATE_CONFIGS[currentMultiRateIndex_];
    const auto& targetConfig = MULTI_RATE_CONFIGS[targetIndex];

    std::cout << "Switching input rate: " << currentConfig.inputRate << " Hz ("
              << currentConfig.ratio << "x) -> " << targetConfig.inputRate << " Hz ("
              << targetConfig.ratio << "x)" << std::endl;

    // Select the source FFT for the target configuration
    cufftComplex* sourceFFT = d_filterFFT_Multi_[targetIndex];

    // Use double-buffering for glitch-free switching
    cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_) ? d_filterFFT_B_ : d_filterFFT_A_;

    Utils::checkCudaError(
        cudaMemcpyAsync(backBuffer, sourceFFT,
                        filterFftSize_ * sizeof(cufftComplex),
                        cudaMemcpyDeviceToDevice, stream_),
        "cudaMemcpyAsync input rate switch"
    );

    // Synchronize to ensure copy is complete before switching
    Utils::checkCudaError(
        cudaStreamSynchronize(stream_),
        "cudaStreamSynchronize input rate switch"
    );

    // Atomic swap to the new buffer
    d_activeFilterFFT_ = backBuffer;

    // Update original filter FFT for EQ restoration
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, sourceFFT,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy update originalFilterFFT"
    );

    // Update host coefficients reference
    h_filterCoeffs_ = h_filterCoeffsMulti_[targetIndex];

    // Update state
    currentMultiRateIndex_ = targetIndex;
    currentInputRate_ = inputSampleRate;
    upsampleRatio_ = targetConfig.ratio;
    currentRateFamily_ = targetConfig.family;

    // Clear EQ state (EQ needs to be re-applied for new rate)
    if (eqApplied_) {
        std::cout << "  Note: EQ was applied. It needs to be re-applied for the new rate." << std::endl;
        eqApplied_ = false;
    }

    std::cout << "Input rate switch complete" << std::endl;
    return true;
}

std::vector<int> GPUUpsampler::getSupportedInputRates() {
    std::vector<int> rates;
    rates.reserve(MULTI_RATE_CONFIG_COUNT);
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        rates.push_back(MULTI_RATE_CONFIGS[i].inputRate);
    }
    return rates;
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
                cufftSetStream(fftPlanForward_, stream),
                "cufftSetStream forward (offline)"
            );
            Utils::checkCufftError(
                cufftExecR2C(fftPlanForward_, d_paddedInput, d_inputFFT),
                "cufftExecR2C block"
            );

            // Complex multiplication with pre-computed filter FFT
            // Note: complexMultiplyKernel modifies data in-place, so we operate directly on d_inputFFT
            threadsPerBlock = 256;
            blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;

            complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                d_inputFFT, d_activeFilterFFT_, fftComplexSize
            );

            Utils::checkCudaError(
                cudaGetLastError(),
                "complexMultiplyKernel launch"
            );

            // Inverse FFT (operate on d_inputFFT since we modified it in-place)
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

            // Save overlap for next block: take the last overlapSize_ samples
            // from the padded input buffer (tail = validOutputPerBlock position).
            Utils::checkCudaError(
                cudaMemcpyAsync(overlapBuffer.data(), d_paddedInput + validOutputPerBlock,
                               overlapSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpy overlap from device"
            );

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
    unregisterHostBuffers();

    if (d_filterCoeffs_) cudaFree(d_filterCoeffs_);
    // Free dual-rate filter FFT buffers
    if (d_filterFFT_44k_) cudaFree(d_filterFFT_44k_);
    if (d_filterFFT_48k_) cudaFree(d_filterFFT_48k_);
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

// Streaming mode methods
bool GPUUpsampler::initializeStreaming() {
    if (fftSize_ == 0 || overlapSize_ == 0) {
        std::cerr << "ERROR: GPU resources not initialized. Call initialize() first." << std::endl;
        return false;
    }

    // Calculate valid output per block (samples at output rate that don't overlap)
    //
    // CRITICAL: validOutputPerBlock_ MUST be divisible by upsampleRatio_ to maintain
    // exact input/output sample rate ratio (1:16). Otherwise, cumulative drift causes
    // buffer underflow and audio glitches.
    //
    // Overlap-save theory: validOutput = N - (L-1) = 1048576 - 999999 = 48577
    // But 48577 is not divisible by 16, so we round down to 48576.
    //
    // streamOverlapSize_ uses the EXACT L-1 value (999999) to maintain correct
    // overlap-save semantics. The remaining 1 sample in d_streamPadded_ is zero-padded
    // (already done by cudaMemsetAsync at the start of each block).
    //
    // d_streamPadded_ layout: [overlap 999999 | new 48576 | zero 1] = 1048576
    int idealValidOutput = fftSize_ - overlapSize_;  // = 48577
    validOutputPerBlock_ = (idealValidOutput / upsampleRatio_) * upsampleRatio_;  // = 48576
    streamOverlapSize_ = overlapSize_;  // = L-1 = 999999 (exact, fixes original drift bug)

    fprintf(stderr, "Streaming parameters:\n");
    fprintf(stderr, "  FFT size: %d\n", fftSize_);
    fprintf(stderr, "  Filter overlap (L-1): %d\n", overlapSize_);
    fprintf(stderr, "  Ideal valid output: %d (not divisible by %d)\n", idealValidOutput, upsampleRatio_);
    fprintf(stderr, "  Actual valid output: %d (rounded to multiple of %d)\n", validOutputPerBlock_, upsampleRatio_);
    fprintf(stderr, "  Stream overlap size: %d (exact L-1)\n", streamOverlapSize_);
    fprintf(stderr, "  Zero-padding at end: %d sample(s)\n", fftSize_ - streamOverlapSize_ - validOutputPerBlock_);

    // Calculate input samples needed per block (exact division, no rounding needed)
    streamValidInputPerBlock_ = validOutputPerBlock_ / upsampleRatio_;  // = 3036

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

    // Allocate device-resident overlap buffers (eliminates H↔D transfers in real-time path)
    // These stay on GPU and are updated via D2D copies only
    Utils::checkCudaError(
        cudaMalloc(&d_overlapLeft_, streamOverlapSize_ * sizeof(float)),
        "cudaMalloc device overlap buffer (left)"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_overlapRight_, streamOverlapSize_ * sizeof(float)),
        "cudaMalloc device overlap buffer (right)"
    );

    // Zero-initialize device overlap buffers
    Utils::checkCudaError(
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(float)),
        "cudaMemset device overlap buffer (left)"
    );
    Utils::checkCudaError(
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(float)),
        "cudaMemset device overlap buffer (right)"
    );

    streamInitialized_ = true;

    fprintf(stderr, "[Streaming] Initialized:\n");
    fprintf(stderr, "  - Input samples per block: %zu\n", streamValidInputPerBlock_);
    fprintf(stderr, "  - Output samples per block: %d\n", validOutputPerBlock_);
    fprintf(stderr, "  - Overlap (stream): %d samples\n", streamOverlapSize_);
    fprintf(stderr, "  - GPU streaming buffers pre-allocated\n");
    fprintf(stderr, "  - Device-resident overlap buffers allocated (no H↔D in RT path)\n");

    return true;
}

void GPUUpsampler::resetStreaming() {
    // Reset device-resident overlap buffers (D2D zero, no H↔D)
    if (d_overlapLeft_) {
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(float));
    }
    if (d_overlapRight_) {
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(float));
    }
    fprintf(stderr, "[Streaming] Reset: device overlap buffers cleared\n");
}

bool GPUUpsampler::processStreamBlock(const float* inputData,
                                       size_t inputFrames,
                                       std::vector<float>& outputData,
                                       cudaStream_t stream,
                                       std::vector<float>& streamInputBuffer,
                                       size_t& streamInputAccumulated) {
    try {
        if (!streamInitialized_) {
            std::cerr << "ERROR: Streaming mode not initialized. Call initializeStreaming() first." << std::endl;
            return false;
        }

        // 1. Accumulate input samples (pin once per stream to avoid repeated cudaHostRegister)
        if (streamInputBuffer.empty()) {
            std::cerr << "ERROR: Streaming input buffer not allocated" << std::endl;
            return false;
        }

        registerStreamInputBuffer(streamInputBuffer, stream);

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

        // Use adjusted overlap size for perfect alignment (set in initializeStreaming)
        int adjustedOverlapSize = streamOverlapSize_;

        // 3. Process one block using pre-allocated GPU buffers
        size_t samplesToProcess = streamValidInputPerBlock_;
        // Note: samplesToProcess * upsampleRatio_ may be > validOutputPerBlock_ due to rounding
        // We only use validOutputPerBlock_ samples to stay within FFT buffer bounds

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

        // Select device-resident overlap buffer based on stream (D2D only, no H↔D)
        float* d_overlap = (stream == streamLeft_) ? d_overlapLeft_ :
                           (stream == streamRight_) ? d_overlapRight_ : d_overlapLeft_;

        // Copy overlap from previous block (D2D - no host transfer!)
        if (adjustedOverlapSize > 0) {
            Utils::checkCudaError(
                cudaMemcpyAsync(d_streamPadded_, d_overlap,
                               adjustedOverlapSize * sizeof(float), cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy streaming overlap D2D"
            );
        }

        // Copy only validOutputPerBlock_ samples to stay within FFT buffer bounds
        // (samplesToProcess * upsampleRatio_ may be slightly larger due to rounding)
        Utils::checkCudaError(
            cudaMemcpyAsync(d_streamPadded_ + adjustedOverlapSize, d_streamUpsampled_,
                           validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToDevice, stream),
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
            d_streamInputFFT_, d_activeFilterFFT_, fftComplexSize
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

        // Extract valid output (discard first adjustedOverlapSize samples)
        // The adjusted overlap size ensures perfect alignment with validOutputPerBlock_
        outputData.resize(validOutputPerBlock_);
        ScopedHostPin outputPinned(outputData.data(),
                                   outputData.size() * sizeof(float),
                                   "cudaHostRegister streaming output");
        Utils::checkCudaError(
            cudaMemcpyAsync(outputData.data(), d_streamConvResult_ + adjustedOverlapSize,
                           validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpy streaming output to host"
        );

        // Save overlap for next block (D2D - no host transfer!)
        // CRITICAL: Must save from d_streamPadded_ (input buffer), not d_streamConvResult_ (convolution output)
        // For next iteration, we need the LAST samples from the padded input buffer
        // which corresponds to: d_streamPadded_[validOutputPerBlock_ : fftSize_]
        // These are the "new" samples that will become "old overlap" in the next iteration
        Utils::checkCudaError(
            cudaMemcpyAsync(d_overlap, d_streamPadded_ + validOutputPerBlock_,
                           adjustedOverlapSize * sizeof(float), cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy streaming overlap D2D save"
        );

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

    } catch (const std::exception& e) {
        std::cerr << "Error in processStreamBlock: " << e.what() << std::endl;
        return false;
    }
}

// ========== EQ Support Methods ==========

void GPUUpsampler::restoreOriginalFilter() {
    if (!d_filterFFT_A_ || !d_filterFFT_B_ || !d_originalFilterFFT_ || filterFftSize_ == 0) {
        return;
    }

    try {
        // Ping-pong: write to back buffer, then swap
        cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_)
                                    ? d_filterFFT_B_ : d_filterFFT_A_;

        Utils::checkCudaError(
            cudaMemcpy(backBuffer, d_originalFilterFFT_,
                      filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy restore original filter to back buffer"
        );

        // Synchronize to ensure copy is complete before swapping
        Utils::checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize before swap");

        // Atomic swap: now convolution kernel will use the restored filter
        d_activeFilterFFT_ = backBuffer;

        eqApplied_ = false;
        std::cout << "EQ: Restored original filter (ping-pong)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EQ: Failed to restore: " << e.what() << std::endl;
    }
}

// CUDA kernel for cepstrum causality window with normalization
// Applies: 1/N normalization (cuFFT doesn't normalize IFFT)
// Plus causality: c[0] unchanged, c[1..N/2-1] *= 2, c[N/2] unchanged, c[N/2+1..N-1] = 0
__global__ void applyCausalityWindowKernel(cufftDoubleReal* cepstrum, int fullN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fullN) return;

    double invN = 1.0 / static_cast<double>(fullN);

    // Apply normalization and causality window together
    if (idx == 0 || idx == fullN / 2) {
        // DC and Nyquist: just normalize
        cepstrum[idx] *= invN;
    } else if (idx < fullN / 2) {
        // Positive time: normalize and double
        cepstrum[idx] *= 2.0 * invN;
    } else {
        // Negative time: zero out
        cepstrum[idx] = 0.0;
    }
}

// CUDA kernel to exponentiate complex values
__global__ void exponentiateComplexKernel(cufftDoubleComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double re = data[idx].x;
    double im = data[idx].y;
    double expRe = exp(re);
    data[idx].x = expRe * cos(im);
    data[idx].y = expRe * sin(im);
}

// CUDA kernel to convert double complex to float complex
__global__ void doubleToFloatComplexKernel(cufftComplex* out, const cufftDoubleComplex* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    out[idx].x = static_cast<float>(in[idx].x);
    out[idx].y = static_cast<float>(in[idx].y);
}

// Apply EQ magnitude with minimum phase reconstruction.
// IMPORTANT: This function assumes the original filter is MINIMUM PHASE.
// If the original filter were linear phase, the group delay characteristics would change
// because we reconstruct minimum phase from combined magnitude |H_filter| × |H_eq|.
// This is intentional for this project (see CLAUDE.md - minimum phase is mandatory).
bool GPUUpsampler::applyEqMagnitude(const std::vector<double>& eqMagnitude) {
    if (!d_filterFFT_A_ || !d_filterFFT_B_ || !d_originalFilterFFT_ || filterFftSize_ == 0) {
        std::cerr << "EQ: Filter not initialized" << std::endl;
        return false;
    }

    if (eqMagnitude.size() != filterFftSize_) {
        std::cerr << "EQ: Magnitude size mismatch: expected " << filterFftSize_
                  << ", got " << eqMagnitude.size() << std::endl;
        return false;
    }

    // Verify persistent EQ resources are initialized
    if (!eqPlanD2Z_ || !eqPlanZ2D_ || !d_eqLogMag_ || !d_eqComplexSpec_) {
        std::cerr << "EQ: Persistent EQ resources not initialized" << std::endl;
        return false;
    }

    // Auto-normalization: prevent clipping by normalizing if max boost > 0dB
    std::vector<double> normalizedMagnitude = eqMagnitude;
    double maxMag = *std::max_element(eqMagnitude.begin(), eqMagnitude.end());
    double normalizationFactor = 1.0;

    if (maxMag > 1.0) {
        normalizationFactor = 1.0 / maxMag;
        for (size_t i = 0; i < normalizedMagnitude.size(); ++i) {
            normalizedMagnitude[i] *= normalizationFactor;
        }
        double normDb = 20.0 * std::log10(normalizationFactor);
        std::cout << "EQ: Auto-normalization applied: " << normDb << " dB "
                  << "(max boost was +" << 20.0 * std::log10(maxMag) << " dB)" << std::endl;
    }

    try {
        size_t fullN = fftSize_;
        size_t halfN = filterFftSize_;  // N/2 + 1

        // Step 1: Use host-cached original filter FFT (no D→H copy needed)
        // h_originalFilterFft_ is populated once at initialization

        // Step 2: Compute combined log magnitude = log(|H_original| * |H_eq|)
        // Uses normalizedMagnitude which has auto-normalization applied if needed
        std::vector<double> logMag(fullN);
        for (size_t i = 0; i < halfN; ++i) {
            double origMag = std::sqrt(h_originalFilterFft_[i].x * h_originalFilterFft_[i].x +
                                       h_originalFilterFft_[i].y * h_originalFilterFft_[i].y);
            double combined = origMag * normalizedMagnitude[i];
            if (combined < 1e-30) combined = 1e-30;  // Avoid log(0)
            logMag[i] = std::log(combined);
        }
        // Hermitian symmetry for negative frequencies
        for (size_t i = 1; i < fullN / 2; ++i) {
            logMag[fullN - i] = logMag[i];
        }

        // Step 3: Convert log magnitude to complex spectrum (for C2R transform)
        // We treat log magnitude as purely real spectrum
        // Using persistent d_eqComplexSpec_ buffer
        std::vector<cufftDoubleComplex> logMagComplex(halfN);
        for (size_t i = 0; i < halfN; ++i) {
            logMagComplex[i].x = logMag[i];
            logMagComplex[i].y = 0.0;
        }
        Utils::checkCudaError(
            cudaMemcpy(d_eqComplexSpec_, logMagComplex.data(),
                      halfN * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice),
            "cudaMemcpy logMag to device"
        );

        // Step 4: IFFT to get cepstrum (using persistent eqPlanZ2D_ and d_eqLogMag_)
        if (cufftExecZ2D(eqPlanZ2D_, d_eqComplexSpec_, d_eqLogMag_) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftExecZ2D failed");
        }

        // Step 5: Apply causality window on GPU (includes 1/N normalization)
        int blockSize = 256;
        int numBlocks = (fullN + blockSize - 1) / blockSize;
        applyCausalityWindowKernel<<<numBlocks, blockSize>>>(d_eqLogMag_, fullN);

        // Step 6: FFT back (using persistent eqPlanD2Z_)
        if (cufftExecD2Z(eqPlanD2Z_, d_eqLogMag_, d_eqComplexSpec_) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftExecD2Z failed");
        }

        // Step 7: Exponentiate on GPU
        numBlocks = (halfN + blockSize - 1) / blockSize;
        exponentiateComplexKernel<<<numBlocks, blockSize>>>(d_eqComplexSpec_, halfN);

        // Step 8: Convert to float and upload to back buffer (ping-pong)
        // Determine back buffer (not currently active)
        cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_)
                                    ? d_filterFFT_B_ : d_filterFFT_A_;

        doubleToFloatComplexKernel<<<numBlocks, blockSize>>>(backBuffer, d_eqComplexSpec_, halfN);

        // Single sync point: ensure all GPU operations complete before pointer swap
        Utils::checkCudaError(cudaDeviceSynchronize(), "EQ cudaDeviceSynchronize");

        // Atomic swap: now convolution kernel will use the new EQ'd filter
        d_activeFilterFFT_ = backBuffer;

        eqApplied_ = true;
        std::cout << "EQ: Applied with minimum phase reconstruction (GPU, ping-pong)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        // No cleanup needed - persistent resources remain valid
        std::cerr << "EQ: Failed to apply minimum phase: " << e.what() << std::endl;
        return false;
    }
}

} // namespace ConvolutionEngine
