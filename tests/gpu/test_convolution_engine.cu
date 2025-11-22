/**
 * @file test_convolution_engine.cu
 * @brief GPU integration tests for ConvolutionEngine (requires CUDA GPU)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "convolution_engine.h"

using namespace ConvolutionEngine;

class ConvolutionEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// Test CUDA device availability
TEST_F(ConvolutionEngineTest, CudaDeviceAvailable) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_GT(deviceCount, 0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
}

// Test GPUUpsampler construction
TEST_F(ConvolutionEngineTest, ConstructorDestructor) {
    GPUUpsampler upsampler;
    // Should construct and destruct without error
    SUCCEED();
}

// Test initialization with valid coefficients
TEST_F(ConvolutionEngineTest, InitializeWithCoefficients) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    bool result = upsampler.initialize(coeffPath, 16, 8192);
    EXPECT_TRUE(result);
}

// Test dual-rate initialization
TEST_F(ConvolutionEngineTest, DualRateInitialize) {
    GPUUpsampler upsampler;

    const char* coeff44k = "data/coefficients/filter_44k_2m_min_phase.bin";
    const char* coeff48k = "data/coefficients/filter_48k_2m_min_phase.bin";

    // Check if files exist
    FILE* f44 = fopen(coeff44k, "rb");
    FILE* f48 = fopen(coeff48k, "rb");
    if (f44 == nullptr || f48 == nullptr) {
        if (f44) fclose(f44);
        if (f48) fclose(f48);
        GTEST_SKIP() << "Coefficient files not found";
    }
    fclose(f44);
    fclose(f48);

    bool result = upsampler.initializeDualRate(coeff44k, coeff48k, 16, 8192);
    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isDualRateEnabled());
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
}

// Test rate family switching
TEST_F(ConvolutionEngineTest, SwitchRateFamily) {
    GPUUpsampler upsampler;

    const char* coeff44k = "data/coefficients/filter_44k_2m_min_phase.bin";
    const char* coeff48k = "data/coefficients/filter_48k_2m_min_phase.bin";

    FILE* f44 = fopen(coeff44k, "rb");
    FILE* f48 = fopen(coeff48k, "rb");
    if (f44 == nullptr || f48 == nullptr) {
        if (f44) fclose(f44);
        if (f48) fclose(f48);
        GTEST_SKIP() << "Coefficient files not found";
    }
    fclose(f44);
    fclose(f48);

    ASSERT_TRUE(upsampler.initializeDualRate(coeff44k, coeff48k, 16, 8192));

    // Initial state should be 44k
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);

    // Switch to 48k
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_48K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);

    // Switch back to 44k
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_44K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);

    // Note: switchRateFamily returns true even when already at target family
    // This is tracked in Issue #77 - for now, verify the family remains unchanged
    upsampler.switchRateFamily(RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
}

// Test basic signal processing
TEST_F(ConvolutionEngineTest, ProcessImpulse) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // Create simple impulse input
    const size_t inputFrames = 8192;
    std::vector<float> input(inputFrames, 0.0f);
    input[0] = 1.0f;  // Impulse at t=0

    std::vector<float> output;
    bool result = upsampler.processChannel(input.data(), inputFrames, output);

    EXPECT_TRUE(result);
    // Output should be 16x longer due to upsampling
    EXPECT_EQ(output.size(), inputFrames * 16);
}

// ============================================================
// Stereo Processing Tests
// ============================================================

TEST_F(ConvolutionEngineTest, ProcessStereo) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    const size_t inputFrames = 8192;
    std::vector<float> leftInput(inputFrames, 0.0f);
    std::vector<float> rightInput(inputFrames, 0.0f);

    // Different impulses for L/R
    leftInput[0] = 1.0f;
    rightInput[100] = 1.0f;

    std::vector<float> leftOutput, rightOutput;
    bool result = upsampler.processStereo(
        leftInput.data(), rightInput.data(), inputFrames,
        leftOutput, rightOutput);

    EXPECT_TRUE(result);
    EXPECT_EQ(leftOutput.size(), inputFrames * 16);
    EXPECT_EQ(rightOutput.size(), inputFrames * 16);

    // Outputs should be different (different impulse positions)
    bool outputsDiffer = false;
    for (size_t i = 0; i < leftOutput.size() && !outputsDiffer; ++i) {
        if (std::abs(leftOutput[i] - rightOutput[i]) > 1e-6f) {
            outputsDiffer = true;
        }
    }
    EXPECT_TRUE(outputsDiffer);
}

// ============================================================
// Streaming Mode Tests
// ============================================================

TEST_F(ConvolutionEngineTest, InitializeStreaming) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    bool result = upsampler.initializeStreaming();
    EXPECT_TRUE(result);
}

TEST_F(ConvolutionEngineTest, ResetStreaming) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));
    ASSERT_TRUE(upsampler.initializeStreaming());

    // Should not crash
    upsampler.resetStreaming();
    SUCCEED();
}

// ============================================================
// EQ Tests
// ============================================================

TEST_F(ConvolutionEngineTest, EqInitiallyNotApplied) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    EXPECT_FALSE(upsampler.isEqApplied());
}

TEST_F(ConvolutionEngineTest, ApplyFlatEq) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // Create flat EQ (unity magnitude)
    size_t fftSize = upsampler.getFilterFftSize();
    std::vector<double> eqMagnitude(fftSize, 1.0);

    bool result = upsampler.applyEqMagnitude(eqMagnitude);
    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isEqApplied());
}

TEST_F(ConvolutionEngineTest, RestoreOriginalFilter) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // Apply EQ
    size_t fftSize = upsampler.getFilterFftSize();
    std::vector<double> eqMagnitude(fftSize, 0.5);  // -6dB
    ASSERT_TRUE(upsampler.applyEqMagnitude(eqMagnitude));
    EXPECT_TRUE(upsampler.isEqApplied());

    // Restore
    upsampler.restoreOriginalFilter();
    EXPECT_FALSE(upsampler.isEqApplied());
}

// ============================================================
// Statistics Tests
// ============================================================

TEST_F(ConvolutionEngineTest, StatsInitiallyZero) {
    GPUUpsampler upsampler;

    const auto& stats = upsampler.getStats();
    EXPECT_DOUBLE_EQ(stats.totalProcessingTime, 0.0);
    EXPECT_EQ(stats.framesProcessed, 0u);
}

TEST_F(ConvolutionEngineTest, StatsAfterProcessing) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    const size_t inputFrames = 8192;
    std::vector<float> input(inputFrames, 0.0f);
    input[0] = 1.0f;

    std::vector<float> output;
    upsampler.processChannel(input.data(), inputFrames, output);

    const auto& stats = upsampler.getStats();
    EXPECT_GT(stats.framesProcessed, 0u);
}

TEST_F(ConvolutionEngineTest, ResetStats) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    const size_t inputFrames = 8192;
    std::vector<float> input(inputFrames, 0.0f);
    std::vector<float> output;
    upsampler.processChannel(input.data(), inputFrames, output);

    upsampler.resetStats();
    const auto& stats = upsampler.getStats();
    EXPECT_DOUBLE_EQ(stats.totalProcessingTime, 0.0);
    EXPECT_EQ(stats.framesProcessed, 0u);
}

// ============================================================
// Getter Tests
// ============================================================

TEST_F(ConvolutionEngineTest, GetUpsampleRatio) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    EXPECT_EQ(upsampler.getUpsampleRatio(), 16);
}

TEST_F(ConvolutionEngineTest, GetSampleRates) {
    GPUUpsampler upsampler;

    const char* coeff44k = "data/coefficients/filter_44k_2m_min_phase.bin";
    const char* coeff48k = "data/coefficients/filter_48k_2m_min_phase.bin";

    FILE* f44 = fopen(coeff44k, "rb");
    FILE* f48 = fopen(coeff48k, "rb");
    if (f44 == nullptr || f48 == nullptr) {
        if (f44) fclose(f44);
        if (f48) fclose(f48);
        GTEST_SKIP() << "Coefficient files not found";
    }
    fclose(f44);
    fclose(f48);

    ASSERT_TRUE(upsampler.initializeDualRate(coeff44k, coeff48k, 16, 8192));

    // 44k family
    EXPECT_EQ(upsampler.getInputSampleRate(), 44100);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 705600);

    // Switch to 48k
    upsampler.switchRateFamily(RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getInputSampleRate(), 48000);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 768000);
}

TEST_F(ConvolutionEngineTest, GetFftSizes) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // FFT sizes should be power of 2 related
    size_t filterFftSize = upsampler.getFilterFftSize();
    size_t fullFftSize = upsampler.getFullFftSize();

    EXPECT_GT(filterFftSize, 0u);
    EXPECT_GT(fullFftSize, 0u);
    // filterFftSize should be fullFftSize/2 + 1 (R2C)
    EXPECT_EQ(filterFftSize, fullFftSize / 2 + 1);
}

// ============================================================
// Output Quality Tests
// ============================================================

TEST_F(ConvolutionEngineTest, OutputNotAllZeros) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    const size_t inputFrames = 8192;
    std::vector<float> input(inputFrames, 0.0f);
    input[0] = 1.0f;

    std::vector<float> output;
    upsampler.processChannel(input.data(), inputFrames, output);

    // Output should have non-zero values (filter response to impulse)
    float maxAbs = 0.0f;
    for (float v : output) {
        maxAbs = std::max(maxAbs, std::abs(v));
    }
    EXPECT_GT(maxAbs, 0.0f);
}

TEST_F(ConvolutionEngineTest, OutputFiniteValues) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found: " << coeffPath;
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    const size_t inputFrames = 8192;
    std::vector<float> input(inputFrames, 0.0f);
    input[0] = 1.0f;

    std::vector<float> output;
    upsampler.processChannel(input.data(), inputFrames, output);

    // All values should be finite (no NaN or Inf)
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ============================================================
// Default Sample Rate Test
// ============================================================

TEST_F(ConvolutionEngineTest, DefaultInputSampleRate) {
    EXPECT_EQ(GPUUpsampler::getDefaultInputSampleRate(), 44100);
}
