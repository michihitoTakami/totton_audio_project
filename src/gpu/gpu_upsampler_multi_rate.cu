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

    // Release CPU-side coefficient memory after GPU transfer (Jetson optimization)
    // FFT spectra are now on GPU; time-domain coefficients are no longer needed
    releaseHostCoefficients();

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

    float previousDelay = getCurrentGroupDelay();
    cufftComplex* previousFilter = d_activeFilterFFT_;

    // Select the source FFT for the target family (considering phase type in quad-phase mode)
    cufftComplex* sourceFFT;
    if (quadPhaseEnabled_) {
        // Quad-phase mode: select based on both family and phase type
        if (targetFamily == RateFamily::RATE_44K) {
            sourceFFT = (phaseType_ == PhaseType::Minimum) ? d_filterFFT_44k_ : d_filterFFT_44k_linear_;
        } else {
            sourceFFT = (phaseType_ == PhaseType::Minimum) ? d_filterFFT_48k_ : d_filterFFT_48k_linear_;
        }
    } else {
        // Standard dual-rate mode: minimum phase only
        sourceFFT = (targetFamily == RateFamily::RATE_44K) ? d_filterFFT_44k_ : d_filterFFT_48k_;
    }

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

    float newDelay = 0.0f;
    if (quadPhaseEnabled_) {
        if (targetFamily == RateFamily::RATE_44K) {
            newDelay = (phaseType_ == PhaseType::Minimum) ? filterCentroid44k_ : filterCentroid44kLinear_;
        } else {
            newDelay = (phaseType_ == PhaseType::Minimum) ? filterCentroid48k_ : filterCentroid48kLinear_;
        }
    } else {
        newDelay = (targetFamily == RateFamily::RATE_44K) ? filterCentroid44k_ : filterCentroid48k_;
    }

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

    // Note: h_filterCoeffs_ (time-domain) is no longer updated here.
    // After initialization, host coefficient vectors are released to save memory.
    // EQ computation uses h_originalFilterFft_ (FFT spectrum) directly.

    // Clear EQ state (EQ needs to be re-applied for new rate family)
    if (eqApplied_) {
        std::cout << "  Note: EQ was applied. It needs to be re-applied for the new rate family." << std::endl;
        eqApplied_ = false;
    }

    currentRateFamily_ = targetFamily;
    inputSampleRate_ = getBaseSampleRate(targetFamily);
    std::cout << "Rate family switch complete" << std::endl;
    startPhaseAlignedCrossfade(previousFilter, previousDelay, newDelay);
    return true;
}

bool GPUUpsampler::initializeMultiRate(const std::string& coefficientDir,
                                        int blockSize,
                                        int initialInputRate) {
    blockSize_ = blockSize;
    currentInputRate_ = initialInputRate;
    inputSampleRate_ = initialInputRate;  // Also set inputSampleRate_ for getOutputSampleRate()

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

    // Load filter coefficient files (skip 1x bypass configs - use Dirac delta)
    int loadedCount = 0;
    int maxTaps = 0;

    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        const auto& config = MULTI_RATE_CONFIGS[i];
        std::string familyStr = (config.family == RateFamily::RATE_44K) ? "44k" : "48k";

        // For 1x bypass mode: generate Dirac delta (passthrough filter)
        // This allows EQ to still be applied via convolution
        if (config.ratio == 1) {
            // Use a minimal Dirac delta - will be zero-padded to maxTaps later
            // Single impulse at t=0 passes signal through unchanged
            h_filterCoeffsMulti_[i].resize(1);
            h_filterCoeffsMulti_[i][0] = 1.0f;  // Unity gain impulse
            filterCentroidMulti_[i] = 0.0f;
            std::cout << "  Generated " << familyStr << "_1x: Dirac delta (passthrough for EQ)"
                      << std::endl;
            ++loadedCount;
            continue;
        }

        // Construct filename: e.g., filter_44k_16x_1024_min_phase.bin
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

        filterCentroidMulti_[i] = PhaseAlignment::computeEnergyCentroid(h_filterCoeffsMulti_[i]);
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

    // Release CPU-side coefficient memory after GPU transfer (Jetson optimization)
    // FFT spectra are now on GPU; time-domain coefficients are no longer needed
    releaseHostCoefficients();

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

    // Save current state for phase crossfade and rollback on error
    float previousDelay = getCurrentGroupDelay();
    cufftComplex* previousFilter = d_activeFilterFFT_;

    int savedMultiRateIndex = currentMultiRateIndex_;
    int savedInputRate = currentInputRate_;
    int savedUpsampleRatio = upsampleRatio_;
    RateFamily savedRateFamily = currentRateFamily_;

    // Reset streaming overlap buffers before switching (prevents artifacts from old rate)
    if (streamInitialized_) {
        resetStreaming();
    }

    // Select the source FFT for the target configuration
    cufftComplex* sourceFFT = d_filterFFT_Multi_[targetIndex];

    // Use double-buffering for glitch-free switching
    cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_) ? d_filterFFT_B_ : d_filterFFT_A_;

    // Copy filter FFT with error handling
    cudaError_t err = cudaMemcpyAsync(backBuffer, sourceFFT,
                                      filterFftSize_ * sizeof(cufftComplex),
                                      cudaMemcpyDeviceToDevice, stream_);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to copy filter FFT: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Synchronize to ensure copy is complete before switching
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to synchronize stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Atomic swap to the new buffer
    d_activeFilterFFT_ = backBuffer;

    // Update original filter FFT for EQ restoration (device)
    err = cudaMemcpy(d_originalFilterFFT_, sourceFFT,
                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to update original filter FFT: " << cudaGetErrorString(err) << std::endl;
        // Rollback: restore previous filter
        d_activeFilterFFT_ = (backBuffer == d_filterFFT_A_) ? d_filterFFT_B_ : d_filterFFT_A_;
        return false;
    }

    // Update host cache of original filter FFT for EQ computation
    h_originalFilterFft_.resize(filterFftSize_);
    err = cudaMemcpy(h_originalFilterFft_.data(), sourceFFT,
                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to update host filter FFT cache: " << cudaGetErrorString(err) << std::endl;
        // Rollback: restore previous state
        currentMultiRateIndex_ = savedMultiRateIndex;
        currentInputRate_ = savedInputRate;
        inputSampleRate_ = savedInputRate;
        upsampleRatio_ = savedUpsampleRatio;
        currentRateFamily_ = savedRateFamily;
        d_activeFilterFFT_ = (backBuffer == d_filterFFT_A_) ? d_filterFFT_B_ : d_filterFFT_A_;
        return false;
    }

    // Note: h_filterCoeffs_ and h_filterCoeffsMulti_ (time-domain) are no longer updated here.
    // After initialization, host coefficient vectors are released to save memory.
    // EQ computation uses h_originalFilterFft_ (FFT spectrum) directly.

    // Update state
    currentMultiRateIndex_ = targetIndex;
    currentInputRate_ = inputSampleRate;
    inputSampleRate_ = inputSampleRate;
    upsampleRatio_ = targetConfig.ratio;
    currentRateFamily_ = targetConfig.family;

    // Invalidate streaming buffers (will be re-initialized by daemon after rate switch)
    if (streamInitialized_) {
        std::cout << "  Streaming mode invalidated - buffers will be re-initialized" << std::endl;
        freeStreamingBuffers();
        std::cout << "  Call initializeStreaming() to resume streaming" << std::endl;
    }

    // Clear EQ state (EQ needs to be re-applied for new rate)
    if (eqApplied_) {
        std::cout << "  Note: EQ was applied. It needs to be re-applied for the new rate." << std::endl;
        eqApplied_ = false;
    }

    std::cout << "Input rate switch complete: " << inputSampleRate << " Hz ("
              << targetConfig.ratio << "x -> " << targetConfig.outputRate << " Hz)" << std::endl;
    float newDelay = filterCentroidMulti_[currentMultiRateIndex_];
    startPhaseAlignedCrossfade(previousFilter, previousDelay, newDelay);
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

bool GPUUpsampler::initializeQuadPhase(const std::string& filterCoeffPath44kMin,
                                       const std::string& filterCoeffPath48kMin,
                                       const std::string& filterCoeffPath44kLinear,
                                       const std::string& filterCoeffPath48kLinear,
                                       int upsampleRatio, int blockSize,
                                       RateFamily initialFamily, PhaseType initialPhase) {
    upsampleRatio_ = upsampleRatio;
    blockSize_ = blockSize;
    currentRateFamily_ = initialFamily;
    phaseType_ = initialPhase;
    inputSampleRate_ = getBaseSampleRate(initialFamily);

    std::cout << "Initializing GPU Upsampler (Quad-Phase Mode)..." << std::endl;
    std::cout << "  Upsample Ratio: " << upsampleRatio_ << "x" << std::endl;
    std::cout << "  Block Size: " << blockSize_ << " samples" << std::endl;
    std::cout << "  Initial Rate Family: " << (initialFamily == RateFamily::RATE_44K ? "44.1kHz" : "48kHz")
              << std::endl;
    std::cout << "  Initial Phase Type: " << (initialPhase == PhaseType::Minimum ? "Minimum" : "Linear")
              << std::endl;

    // Load all 4 coefficient files
    std::cout << "Loading coefficient files..." << std::endl;

    // 44.1kHz minimum phase
    std::cout << "  44.1kHz Minimum: " << filterCoeffPath44kMin << std::endl;
    if (!loadFilterCoefficients(filterCoeffPath44kMin)) {
        std::cerr << "Error: Failed to load 44.1kHz minimum phase coefficients" << std::endl;
        return false;
    }
    h_filterCoeffs44k_ = h_filterCoeffs_;
    std::cout << "    " << h_filterCoeffs44k_.size() << " taps" << std::endl;
    filterCentroid44k_ = baseFilterCentroid_;

    // 48kHz minimum phase
    std::cout << "  48kHz Minimum: " << filterCoeffPath48kMin << std::endl;
    if (!loadFilterCoefficients(filterCoeffPath48kMin)) {
        std::cerr << "Error: Failed to load 48kHz minimum phase coefficients" << std::endl;
        return false;
    }
    h_filterCoeffs48k_ = h_filterCoeffs_;
    std::cout << "    " << h_filterCoeffs48k_.size() << " taps" << std::endl;
    filterCentroid48k_ = baseFilterCentroid_;

    // 44.1kHz linear phase
    std::cout << "  44.1kHz Linear: " << filterCoeffPath44kLinear << std::endl;
    if (!loadFilterCoefficients(filterCoeffPath44kLinear)) {
        std::cerr << "Error: Failed to load 44.1kHz linear phase coefficients" << std::endl;
        return false;
    }
    h_filterCoeffs44k_linear_ = h_filterCoeffs_;
    std::cout << "    " << h_filterCoeffs44k_linear_.size() << " taps" << std::endl;
    filterCentroid44kLinear_ = baseFilterCentroid_;

    // 48kHz linear phase
    std::cout << "  48kHz Linear: " << filterCoeffPath48kLinear << std::endl;
    if (!loadFilterCoefficients(filterCoeffPath48kLinear)) {
        std::cerr << "Error: Failed to load 48kHz linear phase coefficients" << std::endl;
        return false;
    }
    h_filterCoeffs48k_linear_ = h_filterCoeffs_;
    std::cout << "    " << h_filterCoeffs48k_linear_.size() << " taps" << std::endl;
    filterCentroid48kLinear_ = baseFilterCentroid_;

    // Verify all 4 coefficient files have the same tap count
    size_t taps44kMin = h_filterCoeffs44k_.size();
    size_t taps48kMin = h_filterCoeffs48k_.size();
    size_t taps44kLinear = h_filterCoeffs44k_linear_.size();
    size_t taps48kLinear = h_filterCoeffs48k_linear_.size();

    if (taps44kMin != taps48kMin || taps44kMin != taps44kLinear || taps44kMin != taps48kLinear) {
        std::cerr << "Error: Coefficient tap counts do not match:" << std::endl;
        std::cerr << "  44k min: " << taps44kMin << ", 48k min: " << taps48kMin
                  << ", 44k linear: " << taps44kLinear << ", 48k linear: " << taps48kLinear << std::endl;
        return false;
    }

    // Use 44k minimum as the primary coefficients for now
    h_filterCoeffs_ = h_filterCoeffs44k_;

    // Setup GPU resources (allocates FFT buffers, plans, etc.)
    if (!setupGPUResources()) {
        std::cerr << "Error: Failed to setup GPU resources" << std::endl;
        return false;
    }

    // Allocate minimum phase FFT buffers (for dual-rate)
    Utils::checkCudaError(cudaMalloc(&d_filterFFT_44k_, filterFftSize_ * sizeof(cufftComplex)),
                          "cudaMalloc d_filterFFT_44k_");
    Utils::checkCudaError(cudaMalloc(&d_filterFFT_48k_, filterFftSize_ * sizeof(cufftComplex)),
                          "cudaMalloc d_filterFFT_48k_");

    // Allocate linear phase FFT buffers
    Utils::checkCudaError(cudaMalloc(&d_filterFFT_44k_linear_, filterFftSize_ * sizeof(cufftComplex)),
                          "cudaMalloc d_filterFFT_44k_linear_");
    Utils::checkCudaError(cudaMalloc(&d_filterFFT_48k_linear_, filterFftSize_ * sizeof(cufftComplex)),
                          "cudaMalloc d_filterFFT_48k_linear_");

    std::cout << "Pre-computing FFT for all 4 filter configurations..." << std::endl;

    // Allocate temporary buffer for FFT computation
    float* d_temp;
    Utils::checkCudaError(cudaMalloc(&d_temp, fftSize_ * sizeof(float)),
                          "cudaMalloc d_temp for quad-phase FFT");

    // 44.1kHz minimum phase
    Utils::checkCudaError(cudaMemset(d_temp, 0, fftSize_ * sizeof(float)), "cudaMemset d_temp 44k min");
    Utils::checkCudaError(cudaMemcpy(d_temp, h_filterCoeffs44k_.data(),
                                     h_filterCoeffs44k_.size() * sizeof(float), cudaMemcpyHostToDevice),
                          "cudaMemcpy filter 44k min");
    Utils::checkCufftError(cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_44k_),
                           "cufftExecR2C filter 44k min");

    // 48kHz minimum phase
    Utils::checkCudaError(cudaMemset(d_temp, 0, fftSize_ * sizeof(float)), "cudaMemset d_temp 48k min");
    Utils::checkCudaError(cudaMemcpy(d_temp, h_filterCoeffs48k_.data(),
                                     h_filterCoeffs48k_.size() * sizeof(float), cudaMemcpyHostToDevice),
                          "cudaMemcpy filter 48k min");
    Utils::checkCufftError(cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_48k_),
                           "cufftExecR2C filter 48k min");

    // 44.1kHz linear phase
    Utils::checkCudaError(cudaMemset(d_temp, 0, fftSize_ * sizeof(float)), "cudaMemset d_temp 44k linear");
    Utils::checkCudaError(cudaMemcpy(d_temp, h_filterCoeffs44k_linear_.data(),
                                     h_filterCoeffs44k_linear_.size() * sizeof(float), cudaMemcpyHostToDevice),
                          "cudaMemcpy filter 44k linear");
    Utils::checkCufftError(cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_44k_linear_),
                           "cufftExecR2C filter 44k linear");

    // 48kHz linear phase
    Utils::checkCudaError(cudaMemset(d_temp, 0, fftSize_ * sizeof(float)), "cudaMemset d_temp 48k linear");
    Utils::checkCudaError(cudaMemcpy(d_temp, h_filterCoeffs48k_linear_.data(),
                                     h_filterCoeffs48k_linear_.size() * sizeof(float), cudaMemcpyHostToDevice),
                          "cudaMemcpy filter 48k linear");
    Utils::checkCufftError(cufftExecR2C(fftPlanForward_, d_temp, d_filterFFT_48k_linear_),
                           "cufftExecR2C filter 48k linear");

    cudaFree(d_temp);
    cudaDeviceSynchronize();

    // Set active filter based on initial family and phase type
    cufftComplex* initialFilter;
    if (initialFamily == RateFamily::RATE_44K) {
        initialFilter =
            (initialPhase == PhaseType::Minimum) ? d_filterFFT_44k_ : d_filterFFT_44k_linear_;
        h_filterCoeffs_ = (initialPhase == PhaseType::Minimum) ? h_filterCoeffs44k_ : h_filterCoeffs44k_linear_;
    } else {
        initialFilter =
            (initialPhase == PhaseType::Minimum) ? d_filterFFT_48k_ : d_filterFFT_48k_linear_;
        h_filterCoeffs_ = (initialPhase == PhaseType::Minimum) ? h_filterCoeffs48k_ : h_filterCoeffs48k_linear_;
    }

    // Copy to original and active filter buffers
    Utils::checkCudaError(cudaMemcpy(d_originalFilterFFT_, initialFilter,
                                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy to originalFilterFFT");

    Utils::checkCudaError(cudaMemcpy(d_filterFFT_A_, initialFilter,
                                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy to filterFFT_A");
    d_activeFilterFFT_ = d_filterFFT_A_;

    // Update host cache of original filter FFT for EQ computation
    h_originalFilterFft_.resize(filterFftSize_);
    Utils::checkCudaError(cudaMemcpy(h_originalFilterFft_.data(), initialFilter,
                                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
                          "cudaMemcpy to h_originalFilterFft_");

    // Release CPU-side coefficient memory after GPU transfer (Jetson optimization)
    // FFT spectra are now on GPU; time-domain coefficients are no longer needed
    releaseHostCoefficients();

    quadPhaseEnabled_ = true;
    dualRateEnabled_ = true;  // Quad-phase implies dual-rate support
    std::cout << "GPU Upsampler (Quad-Phase) initialized successfully!" << std::endl;
    return true;
}

bool GPUUpsampler::switchPhaseType(PhaseType targetPhase) {
    if (!quadPhaseEnabled_) {
        std::cerr << "Error: Quad-phase mode not enabled. Use setPhaseType() for non-quad-phase mode."
                  << std::endl;
        return false;
    }

    if (targetPhase == phaseType_) {
        std::cout << "Already at target phase type" << std::endl;
        return true;
    }

    std::cout << "Switching phase type: " << (phaseType_ == PhaseType::Minimum ? "Minimum" : "Linear")
              << " -> " << (targetPhase == PhaseType::Minimum ? "Minimum" : "Linear") << std::endl;

    float previousDelay = getCurrentGroupDelay();
    cufftComplex* previousFilter = d_activeFilterFFT_;

    // Select the source FFT based on current rate family and target phase
    cufftComplex* sourceFFT;
    if (currentRateFamily_ == RateFamily::RATE_44K) {
        sourceFFT = (targetPhase == PhaseType::Minimum) ? d_filterFFT_44k_ : d_filterFFT_44k_linear_;
        h_filterCoeffs_ = (targetPhase == PhaseType::Minimum) ? h_filterCoeffs44k_ : h_filterCoeffs44k_linear_;
    } else {
        sourceFFT = (targetPhase == PhaseType::Minimum) ? d_filterFFT_48k_ : d_filterFFT_48k_linear_;
        h_filterCoeffs_ = (targetPhase == PhaseType::Minimum) ? h_filterCoeffs48k_ : h_filterCoeffs48k_linear_;
    }

    // Use double-buffering for glitch-free switching
    cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_) ? d_filterFFT_B_ : d_filterFFT_A_;

    Utils::checkCudaError(cudaMemcpyAsync(backBuffer, sourceFFT,
                                          filterFftSize_ * sizeof(cufftComplex),
                                          cudaMemcpyDeviceToDevice, stream_),
                          "cudaMemcpyAsync phase type switch");

    // Synchronize to ensure copy is complete before switching
    Utils::checkCudaError(cudaStreamSynchronize(stream_), "cudaStreamSynchronize phase type switch");

    // Atomic swap to the new buffer
    d_activeFilterFFT_ = backBuffer;

    // Update original filter FFT for EQ restoration
    Utils::checkCudaError(cudaMemcpy(d_originalFilterFFT_, sourceFFT,
                                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy update originalFilterFFT");

    // Update host cache of original filter FFT for EQ computation
    Utils::checkCudaError(cudaMemcpy(h_originalFilterFft_.data(), sourceFFT,
                                     filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
                          "cudaMemcpy update h_originalFilterFft_");

    // Note: h_filterCoeffs_ (time-domain) is no longer updated here.
    // After initialization, host coefficient vectors are released to save memory.
    // EQ computation uses h_originalFilterFft_ (FFT spectrum) directly.

    // Clear EQ state (EQ needs to be re-applied for new phase type)
    if (eqApplied_) {
        std::cout << "  Note: EQ was applied. It needs to be re-applied for the new phase type."
                  << std::endl;
        eqApplied_ = false;
    }

    phaseType_ = targetPhase;
    std::cout << "Phase type switch complete" << std::endl;

    float newDelay = 0.0f;
    if (currentRateFamily_ == RateFamily::RATE_44K) {
        newDelay = (phaseType_ == PhaseType::Minimum) ? filterCentroid44k_ : filterCentroid44kLinear_;
    } else {
        newDelay = (phaseType_ == PhaseType::Minimum) ? filterCentroid48k_ : filterCentroid48kLinear_;
    }
    startPhaseAlignedCrossfade(previousFilter, previousDelay, newDelay);
    return true;
}

}  // namespace ConvolutionEngine
