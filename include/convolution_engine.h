#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include "core/config_loader.h"  // PhaseType enum
#include "gpu/partition_plan.h"
#include "gpu/pinned_allocator.h"
#include "gpu/precision_traits.h"
#include "phase_alignment.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <array>
#include <complex>
#include <string>
#include <vector>

namespace ConvolutionEngine {

using StreamFloatVector = CudaPinnedVector<float>;

// Rate family enumeration for multi-rate support
enum class RateFamily {
    RATE_44K = 0,  // 44.1kHz family (44100, 88200, 176400 Hz)
    RATE_48K = 1,  // 48kHz family (48000, 96000, 192000 Hz)
    RATE_UNKNOWN = -1
};

// Detect rate family from sample rate
inline RateFamily detectRateFamily(int sampleRate) {
    if (sampleRate % 44100 == 0)
        return RateFamily::RATE_44K;
    if (sampleRate % 48000 == 0)
        return RateFamily::RATE_48K;
    return RateFamily::RATE_UNKNOWN;
}

// Get output sample rate for a rate family (16x upsampling)
inline int getOutputSampleRate(RateFamily family) {
    switch (family) {
    case RateFamily::RATE_44K:
        return 705600;
    case RateFamily::RATE_48K:
        return 768000;
    default:
        return 0;
    }
}

// Get base sample rate for a rate family
inline int getBaseSampleRate(RateFamily family) {
    switch (family) {
    case RateFamily::RATE_44K:
        return 44100;
    case RateFamily::RATE_48K:
        return 48000;
    default:
        return 0;
    }
}

// Multi-rate configuration for a single filter
struct MultiRateFilterConfig {
    int inputRate;      // Input sample rate (e.g., 44100, 88200, 176400, 352800)
    int outputRate;     // Output sample rate (705600 for 44k family, 768000 for 48k family)
    int ratio;          // Upsample ratio (16, 8, 4, or 2)
    RateFamily family;  // Rate family (44k or 48k)
};

// Supported multi-rate configurations (10 total: 5 per family)
// Index mapping: 0-4 for 44k family (16x,8x,4x,2x,1x), 5-9 for 48k family (16x,8x,4x,2x,1x)
// 1x = bypass mode (no upsampling, input already at output rate)
constexpr int MULTI_RATE_CONFIG_COUNT = 10;
constexpr MultiRateFilterConfig MULTI_RATE_CONFIGS[MULTI_RATE_CONFIG_COUNT] = {
    // 44.1kHz family -> 705.6kHz output
    {44100, 705600, 16, RateFamily::RATE_44K},
    {88200, 705600, 8, RateFamily::RATE_44K},
    {176400, 705600, 4, RateFamily::RATE_44K},
    {352800, 705600, 2, RateFamily::RATE_44K},
    {705600, 705600, 1, RateFamily::RATE_44K},  // Bypass: already at output rate
    // 48kHz family -> 768kHz output
    {48000, 768000, 16, RateFamily::RATE_48K},
    {96000, 768000, 8, RateFamily::RATE_48K},
    {192000, 768000, 4, RateFamily::RATE_48K},
    {384000, 768000, 2, RateFamily::RATE_48K},
    {768000, 768000, 1, RateFamily::RATE_48K},  // Bypass: already at output rate
};

// Find config index for given input sample rate (-1 if not found)
inline int findMultiRateConfigIndex(int inputSampleRate) {
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        if (MULTI_RATE_CONFIGS[i].inputRate == inputSampleRate) {
            return i;
        }
    }
    return -1;
}

// Get upsample ratio for given input sample rate (0 if not supported)
inline int getUpsampleRatioForInputRate(int inputSampleRate) {
    int idx = findMultiRateConfigIndex(inputSampleRate);
    return (idx >= 0) ? MULTI_RATE_CONFIGS[idx].ratio : 0;
}

class GPUUpsampler {
   public:
    GPUUpsampler();
    ~GPUUpsampler();

    // Initialize with filter coefficients file and block size (single rate family)
    //
    // Parameters:
    //   filterCoeffPath: Path to binary file containing filter coefficients (float32 array)
    //   upsampleRatio: Integer upsampling ratio (e.g., 16 for 44.1kHz -> 705.6kHz)
    //   blockSize: FFT processing block size in samples (default: 8192)
    //              This determines the basic unit for Overlap-Save convolution.
    //              Larger values improve GPU efficiency but increase latency.
    //              Actual FFT size will be next power-of-2 >= (blockSize + filterTaps - 1)
    //
    //              Trade-offs:
    //              - Small (2048-4096): Lower latency, more CPU-GPU transfers, lower throughput
    //              - Medium (8192-16384): Balanced latency and throughput (recommended)
    //              - Large (32768+): Higher throughput, higher latency, more memory usage
    bool initialize(const std::string& filterCoeffPath, int upsampleRatio, int blockSize = 8192);

    // Initialize with dual rate family support (44.1kHz and 48kHz)
    //
    // Parameters:
    //   filterCoeffPath44k: Path to 44.1kHz family filter coefficients
    //   filterCoeffPath48k: Path to 48kHz family filter coefficients
    //   upsampleRatio: Integer upsampling ratio (typically 16)
    //   blockSize: FFT processing block size
    //   initialFamily: Initial rate family to use (default: 44.1kHz)
    //
    // Both coefficient files are loaded and pre-processed (FFT) at initialization.
    // Switching between families is then glitch-free using double buffering.
    bool initializeDualRate(const std::string& filterCoeffPath44k,
                            const std::string& filterCoeffPath48k, int upsampleRatio,
                            int blockSize = 8192, RateFamily initialFamily = RateFamily::RATE_44K);

    // Initialize with full multi-rate support (all 8 filter configurations)
    //
    // Parameters:
    //   coefficientDir: Directory containing filter coefficient files
    //                   Expected files: filter_44k_16x_*.bin, filter_44k_8x_*.bin, etc.
    //   blockSize: FFT processing block size
    //   initialInputRate: Initial input sample rate (default: 44100)
    //
    // All 8 coefficient files are loaded and pre-processed at initialization.
    // Use switchToInputRate() to change the active filter based on input rate.
    bool initializeMultiRate(const std::string& coefficientDir, int blockSize = 8192,
                             int initialInputRate = 44100);

    // Initialize with quad-phase support (2 rate families × 2 phase types)
    //
    // Parameters:
    //   filterCoeffPath44kMin: Path to 44.1kHz minimum phase filter coefficients
    //   filterCoeffPath48kMin: Path to 48kHz minimum phase filter coefficients
    //   filterCoeffPath44kLinear: Path to 44.1kHz linear phase filter coefficients
    //   filterCoeffPath48kLinear: Path to 48kHz linear phase filter coefficients
    //   upsampleRatio: Integer upsampling ratio (typically 16)
    //   blockSize: FFT processing block size
    //   initialFamily: Initial rate family to use (default: 44.1kHz)
    //   initialPhase: Initial phase type to use (default: Minimum)
    //
    // All 4 coefficient files are loaded and pre-processed (FFT) at initialization.
    // Switching between families/phases is glitch-free using pre-computed FFTs.
    bool initializeQuadPhase(const std::string& filterCoeffPath44kMin,
                             const std::string& filterCoeffPath48kMin,
                             const std::string& filterCoeffPath44kLinear,
                             const std::string& filterCoeffPath48kLinear, int upsampleRatio,
                             int blockSize = 8192, RateFamily initialFamily = RateFamily::RATE_44K,
                             PhaseType initialPhase = PhaseType::Minimum);

    // Switch to a different phase type (glitch-free via pre-computed FFTs)
    // Returns true if switch was successful, false if not in quad-phase mode or error
    // Note: This only works in quad-phase mode. In other modes, use setPhaseType() instead.
    bool switchPhaseType(PhaseType targetPhase);

    // Check if quad-phase mode is enabled
    bool isQuadPhaseEnabled() const {
        return quadPhaseEnabled_;
    }

    // Switch to filter appropriate for given input sample rate
    //
    // Parameters:
    //   inputSampleRate: New input sample rate (e.g., 44100, 88200, 96000, etc.)
    //
    // Returns:
    //   true if switch was successful
    //   false if rate not supported or error occurred
    //
    // Supported rates: 44100, 88200, 176400, 352800 (44k family)
    //                  48000, 96000, 192000, 384000 (48k family)
    bool switchToInputRate(int inputSampleRate);

    // Get list of supported input sample rates
    static std::vector<int> getSupportedInputRates();

    // Check if multi-rate mode is enabled
    bool isMultiRateEnabled() const {
        return multiRateEnabled_;
    }

    // Get current input sample rate (multi-rate mode)
    int getCurrentInputRate() const {
        return currentInputRate_;
    }

    // Switch to a different rate family (glitch-free via double buffering)
    // Returns true if switch was successful, false if already at target or error
    bool switchRateFamily(RateFamily targetFamily);

    // Get current rate family
    RateFamily getCurrentRateFamily() const {
        return currentRateFamily_;
    }

    // Check if dual-rate mode is enabled
    bool isDualRateEnabled() const {
        return dualRateEnabled_;
    }

    // Process single channel audio (mono)
    bool processChannel(const float* inputData, size_t inputFrames, std::vector<float>& outputData);

    // Process stereo audio (L/R channels separately)
    bool processStereo(const float* leftInput, const float* rightInput, size_t inputFrames,
                       std::vector<float>& leftOutput, std::vector<float>& rightOutput);

    // Initialize streaming mode for real-time processing
    // This pre-allocates buffers and prepares the engine for incremental processing
    bool initializeStreaming();

    // Reset streaming state (clears accumulated input and overlap buffers)
    void resetStreaming();

    // Configure partitioned convolution (Issue #351)
    void setPartitionedConvolutionConfig(const AppConfig::PartitionedConvolutionConfig& config);
    bool isPartitionedConvolutionEnabled() const {
        return partitionPlan_.enabled;
    }
    const PartitionPlan& getPartitionPlan() const {
        return partitionPlan_;
    }

    // Free streaming buffers (releases GPU memory)
    // Call this before rate switch or shutdown to free streaming resources
    void freeStreamingBuffers();

    // Process streaming audio block (real-time mode)
    // Accumulates input samples and processes when enough data is available
    // Returns true if output was generated, false if still accumulating
    bool processStreamBlock(const float* inputData, size_t inputFrames,
                            StreamFloatVector& outputData, cudaStream_t stream,
                            StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated);

    // Get performance statistics
    struct Stats {
        double totalProcessingTime;  // seconds
        double gpuUtilization;       // percentage
        size_t framesProcessed;
    };

    const Stats& getStats() const {
        return stats_;
    }
    void resetStats() {
        stats_ = Stats();
    }

    // ========== Phase Type Support ==========

    // Set the phase type for EQ processing
    // - Minimum: EQ uses minimum phase reconstruction (no pre-ringing)
    // - Linear: EQ applies magnitude only, preserving original filter phase
    void setPhaseType(PhaseType type) {
        phaseType_ = type;
    }

    // Get current phase type
    PhaseType getPhaseType() const {
        return phaseType_;
    }

    // Get latency in samples for current phase type
    // - Minimum phase: 0 (causal, no pre-delay)
    // - Linear phase: (filterTaps - 1) / 2
    int getLatencySamples() const {
        if (phaseType_ == PhaseType::Linear) {
            return (filterTaps_ - 1) / 2;
        }
        return 0;
    }

    // Get latency in seconds for current phase type
    double getLatencySeconds() const {
        int outputRate = getOutputSampleRate();
        if (outputRate <= 0)
            return 0.0;
        return static_cast<double>(getLatencySamples()) / outputRate;
    }

    // ========== EQ Support ==========

    // Apply EQ magnitude (dispatches to appropriate method based on phase type)
    // - Minimum phase: full cepstrum-based minimum phase reconstruction
    // - Linear phase: magnitude-only multiplication, preserving filter phase
    bool applyEqMagnitude(const std::vector<double>& eqMagnitude);

    // Apply EQ magnitude only (for linear phase filters)
    // Multiplies filter magnitude by EQ magnitude, preserves original phase
    // Use this when phaseType_ == PhaseType::Linear
    bool applyEqMagnitudeOnly(const std::vector<double>& eqMagnitude);

   private:
    // Internal: Apply EQ with minimum phase reconstruction
    bool applyEqMagnitudeMinPhase(const std::vector<double>& eqMagnitude);

   public:
    // Restore original filter (remove EQ)
    void restoreOriginalFilter();

    // Check if EQ is currently applied
    bool isEqApplied() const {
        return eqApplied_;
    }

    // Get filter FFT size (R2C output size = N/2+1, for computing EQ response)
    size_t getFilterFftSize() const {
        return filterFftSize_;
    }

    // Get full FFT size (N, for frequency calculation)
    size_t getFullFftSize() const {
        return fftSize_;
    }

    // Get upsample ratio
    int getUpsampleRatio() const {
        return upsampleRatio_;
    }

    // Set input sample rate (updates rate family automatically)
    // Call this after initialize() for single-rate mode to get correct latency calculation
    void setInputSampleRate(int sampleRate) {
        inputSampleRate_ = sampleRate;
        currentRateFamily_ = detectRateFamily(sampleRate);
    }

    // Get input sample rate
    int getInputSampleRate() const {
        return inputSampleRate_;
    }

    // Get output sample rate (input rate * upsample ratio)
    int getOutputSampleRate() const {
        return inputSampleRate_ * upsampleRatio_;
    }

    bool isPhaseCrossfadeActive() const {
        return phaseCrossfade_.active;
    }

    // Legacy: Get default input sample rate (44.1kHz)
    static constexpr int getDefaultInputSampleRate() {
        return 44100;
    }

    // Get streaming buffer requirements (for daemon buffer allocation)
    // Returns the number of input samples needed per processing block
    // Use this to size input accumulation buffers (recommend 2x for safety margin)
    size_t getStreamValidInputPerBlock() const {
        return streamValidInputPerBlock_;
    }

    // CUDA streams for async operations (public for daemon access)
    cudaStream_t stream_;       // Primary stream for mono
    cudaStream_t streamLeft_;   // Left channel for stereo parallel
    cudaStream_t streamRight_;  // Right channel for stereo parallel

   private:
    // Load filter coefficients from binary file
    bool loadFilterCoefficients(const std::string& path);

    // Setup CUDA memory and cuFFT plans
    bool setupGPUResources();
    void resizeOverlapBuffers(size_t newSize);

    // Perform Overlap-Save FFT convolution
    bool overlapSaveConvolution(const float* input, size_t inputLength, std::vector<float>& output);

    // Internal helper for processing a channel with a specific stream
    bool processChannelWithStream(const float* inputData, size_t inputFrames,
                                  std::vector<float>& outputData, cudaStream_t stream,
                                  std::vector<float>& overlapBuffer);

    // Free all GPU resources
    void cleanup();

    // Downconversion helpers (float64 -> float32 host)
    float* getOutputScratch(cudaStream_t stream);
    cudaError_t downconvertToHost(float* hostDst, const DeviceSample* deviceSrc, size_t count,
                                  cudaStream_t stream);
    cudaError_t downconvertToHostSync(float* hostDst, const DeviceSample* deviceSrc, size_t count);

    // Host(float32) -> Device(active precision) transfer helper.
    // In float64 builds, avoids passing short-lived host buffers to cudaMemcpyAsync.
    cudaError_t copyHostToDeviceSamplesConvertedAsync(DeviceSample* dst, const float* src,
                                                      size_t count, cudaStream_t stream);

    // Release CPU-side filter coefficient memory after GPU transfer
    // This saves ~100MB of RAM, especially important for Jetson Unified Memory
    // Call this after all GPU transfers are complete (FFT pre-computation done)
    void releaseHostCoefficients();

    void startPhaseAlignedCrossfade(DeviceFftComplex* previousFilter, float previousDelay,
                                    float newDelay);
    void cancelPhaseAlignedCrossfade();
    void applyPhaseAlignedCrossfade(StreamFloatVector& newOutput,
                                    const std::vector<float>& oldOutput, bool advanceProgress);
    int getPhaseCrossfadeSamples() const;
    float getCurrentGroupDelay() const;

    // Configuration
    int upsampleRatio_;
    int blockSize_;
    int filterTaps_;
    int fftSize_;                               // Pre-computed FFT size
    int inputSampleRate_ = 44100;               // Input sample rate (default: 44.1kHz)
    PhaseType phaseType_ = PhaseType::Minimum;  // Filter phase type (default: Minimum)

    // Filter coefficients (single-rate mode)
    std::vector<float> h_filterCoeffs_;              // Host (float32 source file)
    std::vector<DeviceSample> h_filterCoeffsTyped_;  // Host in active precision
    DeviceSample* d_filterCoeffs_;                   // Device

    // Dual-rate support
    bool dualRateEnabled_;          // True if dual-rate mode is active
    RateFamily currentRateFamily_;  // Currently active rate family

    // 44.1kHz family coefficients
    std::vector<float> h_filterCoeffs44k_;  // Host coefficients for 44.1kHz family
    DeviceFftComplex* d_filterFFT_44k_;     // Pre-computed FFT for 44.1kHz family

    // 48kHz family coefficients
    std::vector<float> h_filterCoeffs48k_;  // Host coefficients for 48kHz family
    DeviceFftComplex* d_filterFFT_48k_;     // Pre-computed FFT for 48kHz family

    // Quad-phase support (2 rate families × 2 phase types)
    bool quadPhaseEnabled_;  // True if quad-phase mode is active

    // Linear phase coefficients (44.1kHz and 48kHz)
    std::vector<float> h_filterCoeffs44k_linear_;  // Host coefficients for 44.1kHz linear phase
    std::vector<float> h_filterCoeffs48k_linear_;  // Host coefficients for 48kHz linear phase
    DeviceFftComplex* d_filterFFT_44k_linear_;     // Pre-computed FFT for 44.1kHz linear phase
    DeviceFftComplex* d_filterFFT_48k_linear_;     // Pre-computed FFT for 48kHz linear phase

    // Multi-rate support (8 configurations)
    bool multiRateEnabled_;      // True if multi-rate mode is active
    int currentInputRate_;       // Current input sample rate
    int currentMultiRateIndex_;  // Index into MULTI_RATE_CONFIGS
    std::vector<float>
        h_filterCoeffsMulti_[MULTI_RATE_CONFIG_COUNT];  // Host coefficients for all 8 configs
    DeviceFftComplex*
        d_filterFFT_Multi_[MULTI_RATE_CONFIG_COUNT];  // Pre-computed FFT for all 8 configs

    // Double-buffered filter FFT (ping-pong) for glitch-free EQ updates
    DeviceFftComplex* d_filterFFT_A_;        // Filter FFT buffer A
    DeviceFftComplex* d_filterFFT_B_;        // Filter FFT buffer B
    DeviceFftComplex* d_activeFilterFFT_;    // Currently active filter (points to A or B)
    DeviceFftComplex* d_originalFilterFFT_;  // Original filter FFT (without EQ, for restoration)
    DeviceFftComplex*
        d_crossfadeFilterSnapshot_;  // Snapshot of previous filter during phase crossfade
    size_t filterFftSize_;           // Size of filter FFT arrays
    bool eqApplied_;                 // True if EQ has been applied

    // Working buffers
    DeviceSample* d_inputBlock_;      // Device input block
    DeviceSample* d_outputBlock_;     // Device output block (upsampled)
    DeviceFftComplex* d_inputFFT_;    // FFT of input
    DeviceFftComplex* d_convResult_;  // Convolution result in frequency domain

    // cuFFT plans
    cufftHandle fftPlanForward_;
    cufftHandle fftPlanInverse_;

    // EQ-specific resources (persistent to avoid allocation during real-time EQ switching)
    cufftHandle eqPlanD2Z_;                              // Double-precision R2C for EQ
    cufftHandle eqPlanZ2D_;                              // Double-precision C2R for EQ
    cufftDoubleReal* d_eqLogMag_;                        // GPU buffer for log magnitude (reused)
    cufftDoubleComplex* d_eqComplexSpec_;                // GPU buffer for complex spectrum (reused)
    std::vector<DeviceFftComplex> h_originalFilterFft_;  // Host cache of original filter FFT

    // Statistics
    Stats stats_;

    // State for Overlap-Save
    std::vector<float> overlapBuffer_;       // Store overlap from previous block (mono/left)
    std::vector<float> overlapBufferRight_;  // Store overlap for right channel
    int overlapSize_;

    // Streaming state
    size_t streamValidInputPerBlock_;  // Input samples needed per block (at input rate)
    bool streamInitialized_;           // Whether streaming mode is initialized
    int validOutputPerBlock_;          // Valid output samples per block (after FFT convolution)
    int streamOverlapSize_;            // Adjusted overlap per block for streaming alignment

    // Streaming GPU buffers (pre-allocated to avoid malloc/free in callbacks)
    DeviceSample* d_streamInput_;               // Device buffer for accumulated input
    DeviceSample* d_streamUpsampled_;           // Device buffer for upsampled input
    DeviceSample* d_streamPadded_;              // Device buffer for [overlap | new] concatenation
    DeviceFftComplex* d_streamInputFFT_;        // FFT of padded input
    DeviceSample* d_streamConvResult_;          // Convolution result
    DeviceFftComplex* d_streamInputFFTBackup_;  // Backup for phase-aware crossfade
    DeviceSample* d_streamConvResultOld_;       // Old filter convolution result during crossfade
    float* d_outputScratch_;                    // Primary/mono stream scratch (float output)
    float* d_outputScratchLeft_;                // Left channel scratch (float output)
    float* d_outputScratchRight_;               // Right channel scratch (float output)

    // Device-resident overlap buffers (eliminates H↔D transfers in real-time path)
    DeviceSample* d_overlapLeft_;   // GPU overlap buffer for left channel
    DeviceSample* d_overlapRight_;  // GPU overlap buffer for right channel

    // Host pinned buffers (long-lived buffers use manual register/unregister;
    // temporary outputs use ScopedHostPin inside implementations)
    struct PinnedHostBuffer {
        void* ptr;
        size_t bytes;
    };
    std::vector<PinnedHostBuffer> pinnedHostBuffers_;

    void* pinnedStreamInputLeft_;
    void* pinnedStreamInputRight_;
    void* pinnedStreamInputMono_;
    size_t pinnedStreamInputLeftBytes_;
    size_t pinnedStreamInputRightBytes_;
    size_t pinnedStreamInputMonoBytes_;
    void* pinnedStreamOutputLeft_;
    void* pinnedStreamOutputRight_;
    void* pinnedStreamOutputMono_;
    size_t pinnedStreamOutputLeftBytes_;
    size_t pinnedStreamOutputRightBytes_;
    size_t pinnedStreamOutputMonoBytes_;

    struct PartitionState {
        PartitionDescriptor descriptor;
        int validOutput = 0;
        int overlapSize = 0;
        size_t fftComplexSize = 0;
        int64_t sampleOffset = 0;
        DeviceFftComplex* d_filterFFT[2] = {nullptr, nullptr};
        int activeFilterIndex = 0;

        // Runtime buffers (allocated when streaming is enabled)
        DeviceSample* d_timeDomain = nullptr;
        DeviceFftComplex* d_inputFFT = nullptr;
        DeviceSample* d_overlapLeft = nullptr;
        DeviceSample* d_overlapRight = nullptr;
        cufftHandle planForward = 0;
        cufftHandle planInverse = 0;
    };

    AppConfig::PartitionedConvolutionConfig partitionConfig_;
    PartitionPlan partitionPlan_;
    std::vector<PartitionState> partitionStates_;
    size_t partitionFastIndex_ = 0;
    size_t maxPartitionValidOutput_ = 0;
    int partitionFastFftSize_ = 0;
    int partitionFastFftComplexSize_ = 0;
    DeviceSample* d_tailAccumulator_ = nullptr;
    DeviceSample* d_tailMixBuffer_ = nullptr;
    DeviceSample* d_upsampledHistory_ = nullptr;
    size_t tailAccumulatorSize_ = 0;
    size_t historyBufferSize_ = 0;
    int64_t tailBaseSample_ = 0;
    size_t tailBaseIndex_ = 0;
    int64_t partitionProcessedSamples_ = 0;
    int64_t partitionOutputSamples_ = 0;
    size_t historyWriteIndex_ = 0;
    bool partitionStreamingInitialized_ = false;

    bool setupPartitionStates();
    void freePartitionStates();
    bool initializePartitionedStreaming();
    void resetPartitionedStreaming();
    bool processPartitionedStreamBlock(const float* inputData, size_t inputFrames,
                                       StreamFloatVector& outputData, cudaStream_t stream,
                                       StreamFloatVector& streamInputBuffer,
                                       size_t& streamInputAccumulated);
    bool processPartitionBlock(PartitionState& state, cudaStream_t stream,
                               const DeviceSample* d_newSamples, int newSamples,
                               DeviceSample* d_channelOverlap, StreamFloatVector& tempOutput,
                               StreamFloatVector& outputData);
    void setActiveHostCoefficients(const std::vector<float>& source);
    bool updateActiveImpulseFromSpectrum(const DeviceFftComplex* spectrum,
                                         std::vector<float>& destination);
    bool refreshPartitionFiltersFromHost();
    bool refreshPartitionFiltersFromActiveSpectrum();

    struct PhaseCrossfadeState {
        bool active = false;
        int samplesRemaining = 0;
        int totalSamples = 0;
        int samplesProcessed = 0;
        DeviceFftComplex* previousFilter = nullptr;
        float previousDelay = 0.0f;
        float newDelay = 0.0f;
        bool delayNew = true;
        PhaseAlignment::FractionalDelayLine delayOld;
        PhaseAlignment::FractionalDelayLine delayNewLine;
    };

    PhaseCrossfadeState phaseCrossfade_;
    std::vector<float> crossfadeOldOutput_;
    std::vector<float> crossfadeAlignedOld_;
    std::vector<float> crossfadeAlignedNew_;

    float baseFilterCentroid_ = 0.0f;
    float filterCentroid44k_ = 0.0f;
    float filterCentroid48k_ = 0.0f;
    float filterCentroid44kLinear_ = 0.0f;
    float filterCentroid48kLinear_ = 0.0f;
    std::array<float, MULTI_RATE_CONFIG_COUNT> filterCentroidMulti_{};

    void registerHostBuffer(void* ptr, size_t bytes, const char* context);
    void registerStreamInputBuffer(StreamFloatVector& buffer, cudaStream_t stream);
    void registerStreamOutputBuffer(StreamFloatVector& buffer, cudaStream_t stream);
    void removePinnedHostBuffer(void* ptr);
    void unregisterHostBuffers();
    cufftHandle partitionImpulsePlanInverse_ = 0;
    DeviceSample* d_partitionImpulse_ = nullptr;
};

// Utility functions
namespace Utils {
// Check CUDA errors
void checkCudaError(cudaError_t error, const char* context);

// Check cuFFT errors
void checkCufftError(cufftResult result, const char* context);

// Zero-pad input signal for upsampling
void zeroPad(const float* input, float* output, size_t inputLength, int upsampleRatio);

// Measure GPU utilization (requires nvidia-ml)
double getGPUUtilization();
}  // namespace Utils

}  // namespace ConvolutionEngine

#endif  // CONVOLUTION_ENGINE_H
