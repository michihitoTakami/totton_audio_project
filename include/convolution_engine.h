#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>

namespace ConvolutionEngine {

class GPUUpsampler {
public:
    GPUUpsampler();
    ~GPUUpsampler();

    // Initialize with filter coefficients file and block size
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

    // Get performance statistics
    struct Stats {
        double totalProcessingTime;  // seconds
        double gpuUtilization;       // percentage
        size_t framesProcessed;
    };

    const Stats& getStats() const { return stats_; }
    void resetStats() { stats_ = Stats(); }

private:
    // Load filter coefficients from binary file
    bool loadFilterCoefficients(const std::string& path);

    // Setup CUDA memory and cuFFT plans
    bool setupGPUResources();

    // Perform Overlap-Save FFT convolution
    bool overlapSaveConvolution(const float* input,
                               size_t inputLength,
                               std::vector<float>& output);

    // Free all GPU resources
    void cleanup();

    // Configuration
    int upsampleRatio_;
    int blockSize_;
    int filterTaps_;

    // Filter coefficients
    std::vector<float> h_filterCoeffs_;  // Host
    float* d_filterCoeffs_;              // Device
    cufftComplex* d_filterFFT_;          // Pre-computed filter FFT

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
    std::vector<float> overlapBuffer_;   // Store overlap from previous block
    int overlapSize_;

    // CUDA stream for async operations
    cudaStream_t stream_;
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
