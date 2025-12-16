#ifndef CROSSFEED_ENGINE_H
#define CROSSFEED_ENGINE_H

#include "hrtf/woodworth_model.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <string>
#include <vector>

namespace CrossfeedEngine {

// Head size categories matching HRTF filter files
enum class HeadSize {
    S = 0,   // Small
    M = 1,   // Medium
    L = 2,   // Large
    XL = 3,  // Extra Large
    COUNT = 4
};

// Convert HeadSize to string for file paths
inline const char* headSizeToString(HeadSize size) {
    switch (size) {
    case HeadSize::S:
        return "s";
    case HeadSize::M:
        return "m";
    case HeadSize::L:
        return "l";
    case HeadSize::XL:
        return "xl";
    default:
        return "m";
    }
}

// Convert string to HeadSize (for API commands)
inline HeadSize stringToHeadSize(const std::string& str) {
    if (str == "xs" || str == "XS") {
        // Note: xs is not in enum, map to S for now
        return HeadSize::S;
    } else if (str == "s" || str == "S") {
        return HeadSize::S;
    } else if (str == "m" || str == "M") {
        return HeadSize::M;
    } else if (str == "l" || str == "L") {
        return HeadSize::L;
    } else if (str == "xl" || str == "XL") {
        return HeadSize::XL;
    }
    return HeadSize::M;  // Default to M
}

// Rate family (shared with ConvolutionEngine)
enum class RateFamily {
    RATE_44K = 0,  // 44.1kHz family (705.6kHz output)
    RATE_48K = 1,  // 48kHz family (768kHz output)
    RATE_UNKNOWN = -1
};

// Get rate family string for file paths
inline const char* rateFamilyToString(RateFamily family) {
    switch (family) {
    case RateFamily::RATE_44K:
        return "44k";
    case RateFamily::RATE_48K:
        return "48k";
    default:
        return "44k";
    }
}

// Upsampled output rate for the given family (matches convolution engine)
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

// HRTF metadata from JSON file
struct HRTFMetadata {
    std::string description;
    std::string sizeCategory;
    std::string subjectId;
    int sampleRate;
    std::string rateFamily;
    int nTaps;
    int nChannels;
    std::vector<std::string> channelOrder;  // ["LL", "LR", "RL", "RR"]
    std::string phaseType;                  // "original" for HRTF
    std::string normalization;              // "ild_preserving"
    float maxDcGain;
    float sourceAzimuthLeft;
    float sourceAzimuthRight;
    float sourceElevation;
    std::string license;
    std::string attribution;
    std::string source;
    std::string storageFormat;
};

// 4-channel HRTF FIR filter set
// Channel layout:
//   LL - Left input -> Left output (ipsilateral)
//   LR - Left input -> Right output (contralateral)
//   RL - Right input -> Left output (contralateral)
//   RR - Right input -> Right output (ipsilateral)
//
// Crossfeed formula:
//   Out_L = In_L * LL + In_R * RL
//   Out_R = In_L * LR + In_R * RR
class HRTFProcessor {
   public:
    HRTFProcessor();
    ~HRTFProcessor();

    // Initialize with HRTF data directory and block size
    //
    // Parameters:
    //   hrtfDir: Directory containing HRTF filter files
    //            Expected files: hrtf_{s,m,l,xl}_{44k,48k}.bin/.json
    //   blockSize: FFT processing block size (default: 8192)
    //   initialSize: Initial head size to use (default: M)
    //   initialFamily: Initial rate family (default: 44k)
    //
    // All HRTF files are loaded and pre-processed (FFT) at initialization.
    bool initialize(const std::string& hrtfDir, int blockSize = 8192,
                    HeadSize initialSize = HeadSize::M,
                    RateFamily initialFamily = RateFamily::RATE_44K);

    // Switch to a different head size (glitch-free via ping-pong buffering)
    // Returns true if switch was successful
    bool switchHeadSize(HeadSize targetSize);

    // Switch to a different rate family
    // Returns true if switch was successful
    bool switchRateFamily(RateFamily targetFamily);

    // Get current head size
    HeadSize getCurrentHeadSize() const {
        return currentHeadSize_;
    }

    // Get current rate family
    RateFamily getCurrentRateFamily() const {
        return currentRateFamily_;
    }

    // Enable/disable crossfeed processing
    void setEnabled(bool enabled) {
        enabled_ = enabled;
    }

    bool isEnabled() const {
        return enabled_;
    }

    // Initialize streaming mode for real-time processing
    bool initializeStreaming();

    // Reset streaming state (clears overlap buffers)
    void resetStreaming();

    // Process streaming audio block
    //
    // Parameters:
    //   inputL, inputR: Input audio (already upsampled to 705.6k/768k)
    //   inputFrames: Number of input samples per channel
    //   outputL, outputR: Output vectors (will be resized)
    //   stream: CUDA stream to use
    //   streamInputBufferL, streamInputBufferR: Accumulation buffers
    //   streamInputAccumulatedL, streamInputAccumulatedR: Accumulated counts
    //
    // Returns true if output was generated
    bool processStreamBlock(const float* inputL, const float* inputR, size_t inputFrames,
                            std::vector<float>& outputL, std::vector<float>& outputR,
                            cudaStream_t stream, std::vector<float>& streamInputBufferL,
                            std::vector<float>& streamInputBufferR, size_t& streamInputAccumulatedL,
                            size_t& streamInputAccumulatedR);

    // Process offline (non-streaming) audio
    bool processStereo(const float* inputL, const float* inputR, size_t inputFrames,
                       std::vector<float>& outputL, std::vector<float>& outputR);

    // Get HRTF metadata for current configuration
    const HRTFMetadata& getCurrentMetadata() const;

    // Get streaming buffer requirements
    size_t getStreamValidInputPerBlock() const {
        return streamValidInputPerBlock_;
    }

    size_t getStreamValidOutputPerBlock() const {
        return static_cast<size_t>(validOutputPerBlock_);
    }

    // Allow pinned buffer growth (tests/offline). Daemon should disable to avoid RT realloc.
    void setAllowPinnedBufferGrowth(bool allow) {
        allowPinnedBufferGrowth_ = allow;
    }

    // Get filter tap count
    int getFilterTaps() const {
        return filterTaps_;
    }

    // Get FFT size for filter (N/2+1 complex values)
    // Use this to validate filter size before calling setCombinedFilter
    size_t getFilterFftSize() const {
        return filterFftSize_;
    }

    // Set externally computed combined HRTF filters (already FFT'd)
    //
    // This allows Python/Control Plane to compute combined filters
    // (e.g., HRTF + EQ) and send them to the engine.
    //
    // Parameters:
    //   rateFamily: Target rate family (44k or 48k)
    //   combinedLL, combinedLR, combinedRL, combinedRR: FFT'd filter data (host memory)
    //   filterComplexCount: Number of cufftComplex elements per channel
    //                       Must match getFilterFftSize()
    //
    // Returns: true on success, false if size mismatch or not initialized
    //
    // Thread safety: This method copies data to device memory and updates
    // active filter pointers atomically. Safe to call during processing.
    bool setCombinedFilter(RateFamily rateFamily, const cufftComplex* combinedLL,
                           const cufftComplex* combinedLR, const cufftComplex* combinedRL,
                           const cufftComplex* combinedRR, size_t filterComplexCount);

    bool generateWoodworthProfile(RateFamily rateFamily, float azimuthDeg,
                                  const HRTF::WoodworthParams& params);

    // Check if currently using externally-set combined filter
    bool isUsingCombinedFilter() const {
        return usingCombinedFilter_;
    }

    // Clear combined filter and revert to predefined HRTF
    void clearCombinedFilter();

    // Performance statistics
    struct Stats {
        double totalProcessingTime;
        size_t framesProcessed;
        double gpuUtilization;
    };

    const Stats& getStats() const {
        return stats_;
    }

    void resetStats() {
        stats_ = Stats();
    }

   private:
    // Load HRTF coefficients from binary file
    // expectedTaps: when >0, strictly enforce identical tap counts across all
    // rate families/head sizes to prevent buffer overruns during GPU setup.
    bool loadHRTFCoefficients(const std::string& binPath, const std::string& jsonPath,
                              HeadSize size, RateFamily family, int expectedTaps = -1);

    // Setup GPU resources
    bool setupGPUResources();

    // Compute FFT of loaded filter coefficients
    bool computeFilterFFT(const std::vector<float>& coeffs, cufftComplex* d_filterFFT);

    // Free all GPU resources
    void cleanup();

    // Release CPU-side HRTF coefficient memory after GPU transfer
    // This saves memory, especially important for Jetson Unified Memory
    void releaseHostCoefficients();

    void registerStreamBuffer(std::vector<float>& buffer, void** trackedPtr, size_t* trackedBytes,
                              const char* context);

    // Get filter index for (size, family) combination
    static int getFilterIndex(HeadSize size, RateFamily family) {
        return static_cast<int>(size) * 2 + static_cast<int>(family);
    }

    // Configuration
    int blockSize_;
    int filterTaps_;
    int fftSize_;
    bool enabled_;
    bool initialized_;

    // Current state
    HeadSize currentHeadSize_;
    RateFamily currentRateFamily_;

    // HRTF coefficients (4 sizes * 2 rate families = 8 configs)
    // Each config has 4 channels (LL, LR, RL, RR)
    static constexpr int NUM_CONFIGS = 8;
    static constexpr int NUM_CHANNELS = 4;

    // Host coefficients: [config][channel]
    std::vector<float> h_filterCoeffs_[NUM_CONFIGS][NUM_CHANNELS];

    // Metadata for each config
    HRTFMetadata metadata_[NUM_CONFIGS];

    // Device filter FFT: [config][channel]
    cufftComplex* d_filterFFT_[NUM_CONFIGS][NUM_CHANNELS];

    // Combined filter buffers for externally-set filters
    // [rate_family (0=44k, 1=48k)][channel]
    static constexpr int NUM_RATE_FAMILIES = 2;
    cufftComplex* d_combinedFilterFFT_[NUM_RATE_FAMILIES][NUM_CHANNELS];
    bool combinedFilterLoaded_[NUM_RATE_FAMILIES];  // Track which rate families have combined
                                                    // filters
    bool usingCombinedFilter_;                      // Using combined filter vs predefined

    // Double-buffered active filter FFT (ping-pong) for glitch-free switching
    // [channel] -> points to either config A or config B
    cufftComplex* d_activeFilterFFT_[NUM_CHANNELS];
    int activeFilterConfig_;

    // FFT size for filter
    size_t filterFftSize_;

    // Working buffers (allocated once)
    float* d_inputL_;             // Device input left
    float* d_inputR_;             // Device input right
    float* d_paddedInputL_;       // Padded input for FFT (left)
    float* d_paddedInputR_;       // Padded input for FFT (right)
    cufftComplex* d_inputFFT_L_;  // FFT of left input
    cufftComplex* d_inputFFT_R_;  // FFT of right input
    cufftComplex* d_convLL_;      // Convolution result LL
    cufftComplex* d_convLR_;      // Convolution result LR
    cufftComplex* d_convRL_;      // Convolution result RL
    cufftComplex* d_convRR_;      // Convolution result RR
    float* d_outputL_;            // Output left (LL + RL)
    float* d_outputR_;            // Output right (LR + RR)
    float* d_tempConv_;           // Temporary buffer for IFFT result

    // cuFFT plans
    cufftHandle fftPlanForward_;
    cufftHandle fftPlanInverse_;

    // CUDA streams
    cudaStream_t stream_;
    cudaStream_t streamL_;
    cudaStream_t streamR_;

    // Overlap-Save state
    int overlapSize_;
    std::vector<float> overlapBufferL_;
    std::vector<float> overlapBufferR_;

    // Device-resident overlap buffers (for real-time path)
    float* d_overlapL_;
    float* d_overlapR_;

    // Streaming state
    bool streamInitialized_;
    size_t streamValidInputPerBlock_;
    bool allowPinnedBufferGrowth_;
    void* pinnedStreamInputL_;
    void* pinnedStreamInputR_;
    void* pinnedStreamOutputL_;
    void* pinnedStreamOutputR_;
    size_t pinnedStreamInputLBytes_;
    size_t pinnedStreamInputRBytes_;
    size_t pinnedStreamOutputLBytes_;
    size_t pinnedStreamOutputRBytes_;
    int validOutputPerBlock_;

    // Statistics
    Stats stats_;
};

}  // namespace CrossfeedEngine

#endif  // CROSSFEED_ENGINE_H
