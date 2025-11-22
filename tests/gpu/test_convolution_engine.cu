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

    // Switching to same family should return false
    EXPECT_FALSE(upsampler.switchRateFamily(RateFamily::RATE_44K));
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
