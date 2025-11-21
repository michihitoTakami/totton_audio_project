#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include <vector>
#include <string>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>

namespace ConvolutionEngine {

class GPUUpsampler {
public:
    GPUUpsampler();
    ~GPUUpsampler();

    // Initialize with filter coefficients file and block size
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
    bool initialize(const std::string& filterCoeffPath,
                   int upsampleRatio,
                   int blockSize = 8192);

    // Process single channel audio (mono)
    bool processChannel(const float* inputData,
                       size_t inputFrames,
                       std::vector<float>& outputData);

    // Process stereo audio (L/R channels separately)
    bool processStereo(const float* leftInput,
                      const float* rightInput,
                      size_t inputFrames,
                      std::vector<float>& leftOutput,
                      std::vector<float>& rightOutput);

    // Initialize streaming mode for real-time processing
    // This pre-allocates buffers and prepares the engine for incremental processing
    bool initializeStreaming();

    // Reset streaming state (clears accumulated input and overlap buffers)
    void resetStreaming();

    // Process streaming audio block (real-time mode)
    // Accumulates input samples and processes when enough data is available
    // Returns true if output was generated, false if still accumulating
    bool processStreamBlock(const float* inputData,
                           size_t inputFrames,
                           std::vector<float>& outputData,
                           cudaStream_t stream,
                           std::vector<float>& streamInputBuffer,
                           size_t& streamInputAccumulated);

    // Get performance statistics
    struct Stats {
        double totalProcessingTime;  // seconds
        double gpuUtilization;       // percentage
        size_t framesProcessed;
    };

    const Stats& getStats() const { return stats_; }
    void resetStats() { stats_ = Stats(); }

    // ========== EQ Support ==========

    // Apply EQ frequency response to the filter
    // eqResponse: complex frequency response (same size as filter FFT)
    // The combined filter is stored as: H_combined = H_original * H_eq
    // Call restoreOriginalFilter() to remove EQ
    bool applyEqResponse(const std::vector<std::complex<double>>& eqResponse);

    // Restore original filter (remove EQ)
    void restoreOriginalFilter();

    // Check if EQ is currently applied
    bool isEqApplied() const { return eqApplied_; }

    // Get filter FFT size (R2C output size = N/2+1, for computing EQ response)
    size_t getFilterFftSize() const { return filterFftSize_; }

    // Get full FFT size (N, for frequency calculation)
    size_t getFullFftSize() const { return fftSize_; }

    // Get upsample ratio
    int getUpsampleRatio() const { return upsampleRatio_; }

    // Get input sample rate assumption (for EQ design)
    static constexpr int getDefaultInputSampleRate() { return 44100; }

    // CUDA streams for async operations (public for daemon access)
    cudaStream_t stream_;          // Primary stream for mono
    cudaStream_t streamLeft_;      // Left channel for stereo parallel
    cudaStream_t streamRight_;     // Right channel for stereo parallel

private:
    // Load filter coefficients from binary file
    bool loadFilterCoefficients(const std::string& path);

    // Setup CUDA memory and cuFFT plans
    bool setupGPUResources();
    void resizeOverlapBuffers(size_t newSize);

    // Perform Overlap-Save FFT convolution
    bool overlapSaveConvolution(const float* input,
                               size_t inputLength,
                               std::vector<float>& output);

    // Internal helper for processing a channel with a specific stream
    bool processChannelWithStream(const float* inputData,
                                   size_t inputFrames,
                                   std::vector<float>& outputData,
                                   cudaStream_t stream,
                                   std::vector<float>& overlapBuffer);

    // Free all GPU resources
    void cleanup();

    // Configuration
    int upsampleRatio_;
    int blockSize_;
    int filterTaps_;
    int fftSize_;                        // Pre-computed FFT size

    // Filter coefficients
    std::vector<float> h_filterCoeffs_;  // Host
    float* d_filterCoeffs_;              // Device
    cufftComplex* d_filterFFT_;          // Pre-computed filter FFT (may have EQ applied)
    cufftComplex* d_originalFilterFFT_;  // Original filter FFT (without EQ, for restoration)
    size_t filterFftSize_;               // Size of filter FFT arrays
    bool eqApplied_;                     // True if EQ has been applied to d_filterFFT_

    // Working buffers
    float* d_inputBlock_;                // Device input block
    float* d_outputBlock_;               // Device output block (upsampled)
    cufftComplex* d_inputFFT_;           // FFT of input
    cufftComplex* d_convResult_;         // Convolution result in frequency domain

    // cuFFT plans
    cufftHandle fftPlanForward_;
    cufftHandle fftPlanInverse_;

    // Statistics
    Stats stats_;

    // State for Overlap-Save
    std::vector<float> overlapBuffer_;       // Store overlap from previous block (mono/left)
    std::vector<float> overlapBufferRight_;  // Store overlap for right channel
    int overlapSize_;

    // Streaming state
    size_t streamValidInputPerBlock_;        // Input samples needed per block (at input rate)
    bool streamInitialized_;                 // Whether streaming mode is initialized
    int validOutputPerBlock_;                // Valid output samples per block (after FFT convolution)
    int streamOverlapSize_;                  // Adjusted overlap per block for streaming alignment

    // Streaming GPU buffers (pre-allocated to avoid malloc/free in callbacks)
    float* d_streamInput_;                   // Device buffer for accumulated input
    float* d_streamUpsampled_;               // Device buffer for upsampled input
    float* d_streamPadded_;                  // Device buffer for [overlap | new] concatenation
    cufftComplex* d_streamInputFFT_;         // FFT of padded input
    float* d_streamConvResult_;              // Convolution result

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

    void registerHostBuffer(void* ptr, size_t bytes, const char* context);
    void registerStreamInputBuffer(std::vector<float>& buffer, cudaStream_t stream);
    void removePinnedHostBuffer(void* ptr);
    void unregisterHostBuffers();
};

// Utility functions
namespace Utils {
    // Check CUDA errors
    void checkCudaError(cudaError_t error, const char* context);

    // Check cuFFT errors
    void checkCufftError(cufftResult result, const char* context);

    // Zero-pad input signal for upsampling
    void zeroPad(const float* input, float* output,
                size_t inputLength, int upsampleRatio);

    // Measure GPU utilization (requires nvidia-ml)
    double getGPUUtilization();
}

} // namespace ConvolutionEngine

#endif // CONVOLUTION_ENGINE_H
