#include "convolution_engine.h"
#include "gpu/cuda_utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace ConvolutionEngine {

// GPUUpsampler implementation - Multi-Rate methods

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

    std::cout << "Switching rate family: "
              << (currentRateFamily_ == RateFamily::RATE_44K ? "44.1kHz" : "48kHz")
              << " -> "
              << (targetFamily == RateFamily::RATE_44K ? "44.1kHz" : "48kHz")
              << std::endl;

    // Select the source FFT for the target family
    cufftComplex* sourceFFT = (targetFamily == RateFamily::RATE_44K) ? d_filterFFT_44k_ : d_filterFFT_48k_;

    // Use double-buffering for glitch-free switching
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

    // Update original filter FFT for EQ restoration (device)
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, sourceFFT,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy update originalFilterFFT"
    );

    // Update host cache of original filter FFT for EQ computation
    h_originalFilterFft_.resize(filterFftSize_);
    Utils::checkCudaError(
        cudaMemcpy(h_originalFilterFft_.data(), sourceFFT,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
        "cudaMemcpy update h_originalFilterFft_"
    );

    // Update host coefficients reference
    h_filterCoeffs_ = (targetFamily == RateFamily::RATE_44K) ? h_filterCoeffs44k_ : h_filterCoeffs48k_;

    // Clear EQ state (EQ needs to be re-applied for new rate family)
    if (eqApplied_) {
        std::cout << "  Note: EQ was applied. It needs to be re-applied for the new rate family." << std::endl;
        eqApplied_ = false;
    }

    currentRateFamily_ = targetFamily;
    inputSampleRate_ = getBaseSampleRate(targetFamily);
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
        for (int taps :
             {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2000000}) {
            std::string testPath = filename + std::to_string(taps) + "_min_phase.bin";
            std::ifstream testFile(testPath, std::ios::binary);
            if (testFile.good()) {
                foundPath = testPath;
                break;
            }
        }

        if (foundPath.empty()) {
            std::cerr << "Error: Cannot find coefficient file for " << familyStr << " " << config.ratio
                      << "x in " << coefficientDir << std::endl;
            return false;
        }

        // Load coefficients
        std::ifstream ifs(foundPath, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error: Cannot open coefficient file: " << foundPath << std::endl;
            return false;
        }

        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        int taps = fileSize / sizeof(float);
        h_filterCoeffsMulti_[i].resize(taps);
        ifs.read(reinterpret_cast<char*>(h_filterCoeffsMulti_[i].data()), fileSize);

        if (!ifs) {
            std::cerr << "Error: Failed to read coefficient file: " << foundPath << std::endl;
            return false;
        }

        if (taps > maxTaps) {
            maxTaps = taps;
        }

        std::cout << "  Loaded " << familyStr << "_" << config.ratio << "x: " << taps << " taps ("
                  << fileSize / 1024 << " KB)" << std::endl;
        ++loadedCount;
    }

    if (loadedCount != MULTI_RATE_CONFIG_COUNT) {
        std::cerr << "Error: Failed to load all coefficient files" << std::endl;
        return false;
    }

    // Use the maximum tap count across all filters for buffer sizing
    filterTaps_ = maxTaps;

    // Copy initial config's coefficients and zero-pad to maxTaps
    const auto& initialCoeffs = h_filterCoeffsMulti_[currentMultiRateIndex_];
    h_filterCoeffs_.resize(maxTaps, 0.0f);
    std::copy(initialCoeffs.begin(), initialCoeffs.end(), h_filterCoeffs_.begin());
    std::cout << "  Max filter taps: " << maxTaps << " (used for buffer sizing)" << std::endl;

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

    // Copy to the original filter FFT for EQ restoration (device)
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, d_activeFilterFFT_,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy to originalFilterFFT"
    );

    // Update host cache of original filter FFT for EQ computation
    h_originalFilterFft_.resize(filterFftSize_);
    Utils::checkCudaError(
        cudaMemcpy(h_originalFilterFft_.data(), d_activeFilterFFT_,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
        "cudaMemcpy to h_originalFilterFft_"
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

    // Update original filter FFT for EQ restoration (device)
    Utils::checkCudaError(
        cudaMemcpy(d_originalFilterFFT_, sourceFFT,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
        "cudaMemcpy update originalFilterFFT"
    );

    // Update host cache of original filter FFT for EQ computation
    h_originalFilterFft_.resize(filterFftSize_);
    Utils::checkCudaError(
        cudaMemcpy(h_originalFilterFft_.data(), sourceFFT,
                   filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
        "cudaMemcpy update h_originalFilterFft_"
    );

    // Update host coefficients - zero-pad to filterTaps_
    const auto& targetCoeffs = h_filterCoeffsMulti_[targetIndex];
    h_filterCoeffs_.assign(filterTaps_, 0.0f);
    std::copy(targetCoeffs.begin(), targetCoeffs.end(), h_filterCoeffs_.begin());

    // Update state
    currentMultiRateIndex_ = targetIndex;
    currentInputRate_ = inputSampleRate;
    upsampleRatio_ = targetConfig.ratio;
    currentRateFamily_ = targetConfig.family;

    // Free streaming buffers if initialized
    if (streamInitialized_) {
        std::cout << "  Streaming mode invalidated - freeing buffers" << std::endl;
        if (d_streamInput_) {
            cudaFree(d_streamInput_);
            d_streamInput_ = nullptr;
        }
        if (d_streamUpsampled_) {
            cudaFree(d_streamUpsampled_);
            d_streamUpsampled_ = nullptr;
        }
        if (d_streamPadded_) {
            cudaFree(d_streamPadded_);
            d_streamPadded_ = nullptr;
        }
        if (d_streamInputFFT_) {
            cudaFree(d_streamInputFFT_);
            d_streamInputFFT_ = nullptr;
        }
        if (d_streamConvResult_) {
            cudaFree(d_streamConvResult_);
            d_streamConvResult_ = nullptr;
        }
        if (d_overlapLeft_) {
            cudaFree(d_overlapLeft_);
            d_overlapLeft_ = nullptr;
        }
        if (d_overlapRight_) {
            cudaFree(d_overlapRight_);
            d_overlapRight_ = nullptr;
        }
        streamInitialized_ = false;
        std::cout << "  Call initializeStreaming() to resume streaming" << std::endl;
    }

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

}  // namespace ConvolutionEngine
