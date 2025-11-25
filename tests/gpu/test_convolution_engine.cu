/**
 * @file test_convolution_engine.cu
 * @brief GPU integration tests for ConvolutionEngine (requires CUDA GPU)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include "convolution_engine.h"

using namespace ConvolutionEngine;

// RAII wrapper for temporary coefficient directory
// Automatically cleans up on destruction (handles ASSERT_* early returns)
class TempCoeffDir {
public:
    static constexpr int TEST_TAPS = 1024;

    TempCoeffDir(bool includeLinearPhase = false) : valid_(false) {
        // Generate unique directory name using high-resolution time + random
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 999999);

        path_ = std::filesystem::temp_directory_path().string() +
                "/gpu_os_test_" + std::to_string(now) + "_" + std::to_string(dis(gen));

        std::error_code ec;
        std::filesystem::create_directories(path_, ec);
        if (ec) {
            error_ = "Failed to create temp directory: " + ec.message();
            return;
        }

        // Generate impulse response (delta function) for each configuration
        std::vector<float> impulse(TEST_TAPS, 0.0f);
        impulse[0] = 1.0f;  // Unit impulse at t=0

        // Generate all 10 coefficient files (minimum phase)
        // Issue #238: Added 1x bypass configs
        const char* configs[] = {
            "44k_16x", "44k_8x", "44k_4x", "44k_2x", "44k_1x",
            "48k_16x", "48k_8x", "48k_4x", "48k_2x", "48k_1x"
        };

        for (const char* config : configs) {
            std::string filename = path_ + "/filter_" + config + "_" +
                                   std::to_string(TEST_TAPS) + "_min_phase.bin";
            if (!writeCoeffFile(filename, impulse)) {
                error_ = "Failed to write coefficient file: " + filename;
                cleanup();
                return;
            }
        }

        // Generate linear phase coefficient files if requested (for quad-phase tests)
        if (includeLinearPhase) {
            // Linear phase uses symmetric impulse response
            std::vector<float> linearImpulse(TEST_TAPS, 0.0f);
            linearImpulse[TEST_TAPS / 2] = 1.0f;  // Center impulse for linear phase

            const char* linearConfigs[] = {"44k_16x", "48k_16x"};
            for (const char* config : linearConfigs) {
                std::string filename = path_ + "/filter_" + config + "_" +
                                       std::to_string(TEST_TAPS) + "_linear.bin";
                if (!writeCoeffFile(filename, linearImpulse)) {
                    error_ = "Failed to write linear coefficient file: " + filename;
                    cleanup();
                    return;
                }
            }
        }

        valid_ = true;
    }

    ~TempCoeffDir() {
        cleanup();
    }

    // Non-copyable, non-movable
    TempCoeffDir(const TempCoeffDir&) = delete;
    TempCoeffDir& operator=(const TempCoeffDir&) = delete;

    bool isValid() const { return valid_; }
    const std::string& path() const { return path_; }
    const std::string& error() const { return error_; }

    // Helper methods for quad-phase tests
    std::string getMinPhasePath(const std::string& rateFamily) const {
        return path_ + "/filter_" + rateFamily + "_16x_" + std::to_string(TEST_TAPS) + "_min_phase.bin";
    }

    std::string getLinearPhasePath(const std::string& rateFamily) const {
        return path_ + "/filter_" + rateFamily + "_16x_" + std::to_string(TEST_TAPS) + "_linear.bin";
    }

private:
    void cleanup() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
            // Ignore cleanup errors (best effort)
        }
    }

    static bool writeCoeffFile(const std::string& path, const std::vector<float>& coeffs) {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) {
            return false;
        }
        ofs.write(reinterpret_cast<const char*>(coeffs.data()), coeffs.size() * sizeof(float));
        return ofs.good();
    }

    std::string path_;
    std::string error_;
    bool valid_;
};

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeff44k = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    const char* coeff48k = "data/coefficients/filter_48k_16x_2m_min_phase.bin";

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

    const char* coeff44k = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    const char* coeff48k = "data/coefficients/filter_48k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeff44k = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    const char* coeff48k = "data/coefficients/filter_48k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

// ============================================================
// Multi-Rate Support Tests
// ============================================================

TEST_F(ConvolutionEngineTest, GetSupportedInputRates) {
    auto rates = GPUUpsampler::getSupportedInputRates();

    // Should return 10 rates (5 for 44k family + 5 for 48k family, including 1x bypass)
    EXPECT_EQ(rates.size(), 10u);

    // Check 44k family rates (including 705600 bypass)
    EXPECT_NE(std::find(rates.begin(), rates.end(), 44100), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 88200), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 176400), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 352800), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 705600), rates.end());  // Bypass

    // Check 48k family rates (including 768000 bypass)
    EXPECT_NE(std::find(rates.begin(), rates.end(), 48000), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 96000), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 192000), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 384000), rates.end());
    EXPECT_NE(std::find(rates.begin(), rates.end(), 768000), rates.end());  // Bypass
}

TEST_F(ConvolutionEngineTest, FindMultiRateConfigIndex) {
    // Test valid rates - 44k family (0-4)
    EXPECT_EQ(findMultiRateConfigIndex(44100), 0);
    EXPECT_EQ(findMultiRateConfigIndex(88200), 1);
    EXPECT_EQ(findMultiRateConfigIndex(176400), 2);
    EXPECT_EQ(findMultiRateConfigIndex(352800), 3);
    EXPECT_EQ(findMultiRateConfigIndex(705600), 4);  // 1x bypass

    // Test valid rates - 48k family (5-9)
    EXPECT_EQ(findMultiRateConfigIndex(48000), 5);
    EXPECT_EQ(findMultiRateConfigIndex(96000), 6);
    EXPECT_EQ(findMultiRateConfigIndex(192000), 7);
    EXPECT_EQ(findMultiRateConfigIndex(384000), 8);
    EXPECT_EQ(findMultiRateConfigIndex(768000), 9);  // 1x bypass

    // Test invalid rates
    EXPECT_EQ(findMultiRateConfigIndex(44000), -1);
    EXPECT_EQ(findMultiRateConfigIndex(48001), -1);
    EXPECT_EQ(findMultiRateConfigIndex(0), -1);
}

TEST_F(ConvolutionEngineTest, GetUpsampleRatioForInputRate) {
    // 44k family: 16x, 8x, 4x, 2x, 1x (bypass)
    EXPECT_EQ(getUpsampleRatioForInputRate(44100), 16);
    EXPECT_EQ(getUpsampleRatioForInputRate(88200), 8);
    EXPECT_EQ(getUpsampleRatioForInputRate(176400), 4);
    EXPECT_EQ(getUpsampleRatioForInputRate(352800), 2);
    EXPECT_EQ(getUpsampleRatioForInputRate(705600), 1);  // Issue #238: bypass

    // 48k family: 16x, 8x, 4x, 2x, 1x (bypass)
    EXPECT_EQ(getUpsampleRatioForInputRate(48000), 16);
    EXPECT_EQ(getUpsampleRatioForInputRate(96000), 8);
    EXPECT_EQ(getUpsampleRatioForInputRate(192000), 4);
    EXPECT_EQ(getUpsampleRatioForInputRate(384000), 2);
    EXPECT_EQ(getUpsampleRatioForInputRate(768000), 1);  // Issue #238: bypass

    // Invalid rate
    EXPECT_EQ(getUpsampleRatioForInputRate(44000), 0);
}

TEST_F(ConvolutionEngineTest, MultiRateConfigValues) {
    // Verify MULTI_RATE_CONFIGS values
    // Issue #238: Added 1x bypass configs, now 10 total (5 per family)
    EXPECT_EQ(MULTI_RATE_CONFIG_COUNT, 10);

    // 44k family configs (indices 0-4: 16x, 8x, 4x, 2x, 1x)
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(MULTI_RATE_CONFIGS[i].family, RateFamily::RATE_44K);
        EXPECT_EQ(MULTI_RATE_CONFIGS[i].outputRate, 705600);
    }

    // 48k family configs (indices 5-9: 16x, 8x, 4x, 2x, 1x)
    for (int i = 5; i < 10; ++i) {
        EXPECT_EQ(MULTI_RATE_CONFIGS[i].family, RateFamily::RATE_48K);
        EXPECT_EQ(MULTI_RATE_CONFIGS[i].outputRate, 768000);
    }

    // Verify ratio * inputRate = outputRate
    for (int i = 0; i < MULTI_RATE_CONFIG_COUNT; ++i) {
        const auto& config = MULTI_RATE_CONFIGS[i];
        EXPECT_EQ(config.inputRate * config.ratio, config.outputRate);
    }
}

TEST_F(ConvolutionEngineTest, MultiRateNotEnabledByDefault) {
    GPUUpsampler upsampler;

    // Before initialization, multi-rate should be disabled
    EXPECT_FALSE(upsampler.isMultiRateEnabled());
}

// Test multi-rate initialization (generates coefficients dynamically)
TEST_F(ConvolutionEngineTest, InitializeMultiRate) {
    TempCoeffDir tempDir;  // RAII: auto-cleanup on scope exit
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    bool result = upsampler.initializeMultiRate(tempDir.path(), 8192, 44100);
    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isMultiRateEnabled());
    EXPECT_EQ(upsampler.getCurrentInputRate(), 44100);
    EXPECT_EQ(upsampler.getUpsampleRatio(), 16);
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
}

TEST_F(ConvolutionEngineTest, InitializeMultiRateWith48k) {
    TempCoeffDir tempDir;
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    bool result = upsampler.initializeMultiRate(tempDir.path(), 8192, 48000);
    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isMultiRateEnabled());
    EXPECT_EQ(upsampler.getCurrentInputRate(), 48000);
    EXPECT_EQ(upsampler.getUpsampleRatio(), 16);
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
}

TEST_F(ConvolutionEngineTest, SwitchToInputRate) {
    TempCoeffDir tempDir;
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeMultiRate(tempDir.path(), 8192, 44100));

    // Switch to 88200 (8x)
    EXPECT_TRUE(upsampler.switchToInputRate(88200));
    EXPECT_EQ(upsampler.getCurrentInputRate(), 88200);
    EXPECT_EQ(upsampler.getUpsampleRatio(), 8);
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 88200 * 8);

    // Switch to 48000 (16x, different family)
    EXPECT_TRUE(upsampler.switchToInputRate(48000));
    EXPECT_EQ(upsampler.getCurrentInputRate(), 48000);
    EXPECT_EQ(upsampler.getUpsampleRatio(), 16);
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 48000 * 16);

    // Switch to 192000 (4x)
    EXPECT_TRUE(upsampler.switchToInputRate(192000));
    EXPECT_EQ(upsampler.getCurrentInputRate(), 192000);
    EXPECT_EQ(upsampler.getUpsampleRatio(), 4);
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 192000 * 4);
}

TEST_F(ConvolutionEngineTest, SwitchToSameInputRate) {
    TempCoeffDir tempDir;
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeMultiRate(tempDir.path(), 8192, 44100));

    // Switch to same rate should succeed
    EXPECT_TRUE(upsampler.switchToInputRate(44100));
    EXPECT_EQ(upsampler.getCurrentInputRate(), 44100);
}

TEST_F(ConvolutionEngineTest, SwitchToInvalidInputRate) {
    TempCoeffDir tempDir;
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeMultiRate(tempDir.path(), 8192, 44100));

    // Switch to invalid rate should fail
    EXPECT_FALSE(upsampler.switchToInputRate(44000));
    EXPECT_FALSE(upsampler.switchToInputRate(0));

    // Current rate should remain unchanged
    EXPECT_EQ(upsampler.getCurrentInputRate(), 44100);
}

TEST_F(ConvolutionEngineTest, SwitchToInputRateWithoutMultiRateInit) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    // Initialize with single-rate mode
    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // switchToInputRate should fail when not in multi-rate mode
    EXPECT_FALSE(upsampler.switchToInputRate(48000));
}

// ============================================================
// Phase Type Support Tests
// ============================================================

TEST_F(ConvolutionEngineTest, DefaultPhaseTypeIsMinimum) {
    GPUUpsampler upsampler;
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
}

TEST_F(ConvolutionEngineTest, SetPhaseType) {
    GPUUpsampler upsampler;

    upsampler.setPhaseType(PhaseType::Linear);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    upsampler.setPhaseType(PhaseType::Minimum);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
}

TEST_F(ConvolutionEngineTest, SetInputSampleRate44k) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    upsampler.setInputSampleRate(44100);
    EXPECT_EQ(upsampler.getInputSampleRate(), 44100);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 705600);  // 44100 * 16
}

TEST_F(ConvolutionEngineTest, SetInputSampleRate48k) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    upsampler.setInputSampleRate(48000);
    EXPECT_EQ(upsampler.getInputSampleRate(), 48000);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 768000);  // 48000 * 16
}

TEST_F(ConvolutionEngineTest, LatencyMinimumPhaseIsZero) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    upsampler.setPhaseType(PhaseType::Minimum);
    EXPECT_EQ(upsampler.getLatencySamples(), 0);
    EXPECT_DOUBLE_EQ(upsampler.getLatencySeconds(), 0.0);
}

TEST_F(ConvolutionEngineTest, LatencyLinearPhase) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    upsampler.setInputSampleRate(44100);
    upsampler.setPhaseType(PhaseType::Linear);

    // 2M taps: (2000000 - 1) / 2 = 999999.5 -> 999999 samples
    EXPECT_EQ(upsampler.getLatencySamples(), 999999);

    // 999999 / 705600 ≈ 1.417 seconds
    double expectedLatency = 999999.0 / 705600.0;
    EXPECT_NEAR(upsampler.getLatencySeconds(), expectedLatency, 0.001);
}

TEST_F(ConvolutionEngineTest, LatencyLinearPhase48k) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    upsampler.setInputSampleRate(48000);  // 48k family
    upsampler.setPhaseType(PhaseType::Linear);

    // 2M taps: (2000000 - 1) / 2 = 999999 samples
    EXPECT_EQ(upsampler.getLatencySamples(), 999999);

    // 999999 / 768000 ≈ 1.302 seconds (different from 44k!)
    double expectedLatency = 999999.0 / 768000.0;
    EXPECT_NEAR(upsampler.getLatencySeconds(), expectedLatency, 0.001);
}

TEST_F(ConvolutionEngineTest, ApplyEqLinearPhase) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // Set to linear phase mode
    upsampler.setPhaseType(PhaseType::Linear);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    // Create flat EQ (unity magnitude)
    size_t fftSize = upsampler.getFilterFftSize();
    std::vector<double> eqMagnitude(fftSize, 1.0);

    // Apply EQ - should use applyEqMagnitudeOnly internally
    bool result = upsampler.applyEqMagnitude(eqMagnitude);
    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isEqApplied());
}

TEST_F(ConvolutionEngineTest, ApplyEqLinearPhaseWithBoost) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // Set to linear phase mode
    upsampler.setPhaseType(PhaseType::Linear);

    // Create EQ with boost (should trigger auto-normalization)
    size_t fftSize = upsampler.getFilterFftSize();
    std::vector<double> eqMagnitude(fftSize, 2.0);  // +6dB boost

    // Apply EQ - should auto-normalize
    bool result = upsampler.applyEqMagnitude(eqMagnitude);
    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isEqApplied());
}

TEST_F(ConvolutionEngineTest, RestoreFilterAfterLinearPhaseEq) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));

    // Set to linear phase mode and apply EQ
    upsampler.setPhaseType(PhaseType::Linear);
    size_t fftSize = upsampler.getFilterFftSize();
    std::vector<double> eqMagnitude(fftSize, 0.5);  // -6dB
    ASSERT_TRUE(upsampler.applyEqMagnitude(eqMagnitude));
    EXPECT_TRUE(upsampler.isEqApplied());

    // Restore original filter
    upsampler.restoreOriginalFilter();
    EXPECT_FALSE(upsampler.isEqApplied());
}

// ============================================================
// Quad-Phase Support Tests
// ============================================================

TEST_F(ConvolutionEngineTest, QuadPhaseNotEnabledByDefault) {
    GPUUpsampler upsampler;
    EXPECT_FALSE(upsampler.isQuadPhaseEnabled());
}

TEST_F(ConvolutionEngineTest, InitializeQuadPhase) {
    TempCoeffDir tempDir(true);  // Include linear phase files
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    bool result = upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Minimum);

    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isQuadPhaseEnabled());
    EXPECT_TRUE(upsampler.isDualRateEnabled());  // Quad-phase implies dual-rate
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
}

TEST_F(ConvolutionEngineTest, InitializeQuadPhaseWith48kLinear) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    bool result = upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_48K, PhaseType::Linear);

    EXPECT_TRUE(result);
    EXPECT_TRUE(upsampler.isQuadPhaseEnabled());
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);
}

TEST_F(ConvolutionEngineTest, SwitchPhaseTypeInQuadPhase) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Minimum));

    // Initial state
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);

    // Switch to linear phase
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Linear));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    // Switch back to minimum phase
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Minimum));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);

    // Switch to same phase (should still succeed)
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Minimum));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
}

TEST_F(ConvolutionEngineTest, SwitchPhaseTypeFailsWithoutQuadPhase) {
    GPUUpsampler upsampler;

    const char* coeffPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    FILE* f = fopen(coeffPath, "rb");
    if (f == nullptr) {
        GTEST_SKIP() << "Coefficient file not found";
    }
    fclose(f);

    // Initialize with single-rate mode (not quad-phase)
    ASSERT_TRUE(upsampler.initialize(coeffPath, 16, 8192));
    EXPECT_FALSE(upsampler.isQuadPhaseEnabled());

    // switchPhaseType should fail when not in quad-phase mode
    EXPECT_FALSE(upsampler.switchPhaseType(PhaseType::Linear));
}

TEST_F(ConvolutionEngineTest, SwitchRateFamilyInQuadPhaseMinimum) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Minimum));

    // Initial state
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);

    // Switch to 48k - phase type should remain minimum
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_48K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);

    // Switch back to 44k - phase type should still remain minimum
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_44K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
}

TEST_F(ConvolutionEngineTest, SwitchRateFamilyInQuadPhaseLinear) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Linear));

    // Initial state
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    // Switch to 48k - phase type should remain linear
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_48K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);
}

TEST_F(ConvolutionEngineTest, QuadPhaseCombinedSwitching) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Minimum));

    // Test all 4 combinations
    // 1. 44k Minimum (initial)
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);

    // 2. 44k Linear
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Linear));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    // 3. 48k Linear
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_48K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    // 4. 48k Minimum
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Minimum));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_48K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);

    // Back to 44k Minimum
    EXPECT_TRUE(upsampler.switchRateFamily(RateFamily::RATE_44K));
    EXPECT_EQ(upsampler.getCurrentRateFamily(), RateFamily::RATE_44K);
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
}

TEST_F(ConvolutionEngineTest, QuadPhaseProcessImpulse) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Minimum));

    // Create simple impulse input
    const size_t inputFrames = 8192;
    std::vector<float> input(inputFrames, 0.0f);
    input[0] = 1.0f;  // Impulse at t=0

    std::vector<float> output;
    bool result = upsampler.processChannel(input.data(), inputFrames, output);

    EXPECT_TRUE(result);
    EXPECT_EQ(output.size(), inputFrames * 16);

    // Output should have non-zero values
    float maxAbs = 0.0f;
    for (float v : output) {
        maxAbs = std::max(maxAbs, std::abs(v));
    }
    EXPECT_GT(maxAbs, 0.0f);
}

TEST_F(ConvolutionEngineTest, QuadPhaseLatencyCalculation) {
    TempCoeffDir tempDir(true);
    ASSERT_TRUE(tempDir.isValid()) << tempDir.error();

    GPUUpsampler upsampler;
    ASSERT_TRUE(upsampler.initializeQuadPhase(
        tempDir.getMinPhasePath("44k"),
        tempDir.getMinPhasePath("48k"),
        tempDir.getLinearPhasePath("44k"),
        tempDir.getLinearPhasePath("48k"),
        16, 8192, RateFamily::RATE_44K, PhaseType::Minimum));

    // Minimum phase should have zero latency
    EXPECT_EQ(upsampler.getLatencySamples(), 0);

    // Switch to linear phase - should have non-zero latency
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Linear));
    // Latency = (taps - 1) / 2 = (1024 - 1) / 2 = 511
    EXPECT_EQ(upsampler.getLatencySamples(), 511);
}
