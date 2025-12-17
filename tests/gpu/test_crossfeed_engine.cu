#include "crossfeed_engine.h"

#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
using namespace CrossfeedEngine;

// Test fixture for HRTFProcessor tests
class HRTFProcessorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create a unique test data directory per test process to avoid parallel collisions.
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = "unknown_test";
        if (info) {
            name = std::string(info->test_suite_name()) + "_" + std::string(info->name());
        }
        for (char& c : name) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
                c = '_';
            }
        }
        testDir_ = (std::filesystem::temp_directory_path() /
                    ("test_hrtf_data_" + name + "_" + std::to_string(getpid())))
                       .string();
        std::filesystem::create_directories(testDir_);
    }

    void TearDown() override {
        // Clean up test data
        std::filesystem::remove_all(testDir_);
    }

    // Create synthetic HRTF filter coefficients for testing
    // Simple impulse response that approximates crossfeed
    void createTestHRTFFiles(int taps44k = 256, int taps48k = -1) {
        if (taps48k < 0) {
            taps48k = taps44k;
        }
        // Create 4 channels: LL, LR, RL, RR
        // LL/RR: Strong (ipsilateral) - direct path
        // LR/RL: Weaker (contralateral) - cross path with delay
        std::vector<float> ll;
        std::vector<float> lr;
        std::vector<float> rl;
        std::vector<float> rr;

        // LL: Direct path impulse at t=0
        // (sized per rate family below)

        // LR: Contralateral with ITD delay (~20 samples at 705.6kHz ~ 28us)
        // and ILD attenuation (0.5)
        int itdSamples = 20;
        // (sized per rate family below)

        // RL: Same as LR (symmetric head)
        // (sized per rate family below)

        // RR: Same as LL
        // (sized per rate family below)

        // Write binary file (4 channels interleaved)
        for (const char* size : {"m"}) {
            for (const char* family : {"44k", "48k"}) {
                int taps = (strcmp(family, "44k") == 0) ? taps44k : taps48k;

                ll.assign(taps, 0.0f);
                lr.assign(taps, 0.0f);
                rl.assign(taps, 0.0f);
                rr.assign(taps, 0.0f);

                ll[0] = 1.0f;
                int itdSamples = std::min(taps - 1, 20);
                lr[itdSamples] = 0.5f;
                rl[itdSamples] = 0.5f;
                rr[0] = 1.0f;
                std::string binPath = testDir_ + "/hrtf_" + size + "_" + family + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                binFile.write(reinterpret_cast<const char*>(ll.data()), taps * sizeof(float));
                binFile.write(reinterpret_cast<const char*>(lr.data()), taps * sizeof(float));
                binFile.write(reinterpret_cast<const char*>(rl.data()), taps * sizeof(float));
                binFile.write(reinterpret_cast<const char*>(rr.data()), taps * sizeof(float));
                binFile.close();

                // Write JSON metadata
                int sampleRate = (strcmp(family, "44k") == 0) ? 705600 : 768000;
                json meta = {{"description", "Test HRTF"},
                             {"size_category", size},
                             {"subject_id", "test"},
                             {"sample_rate", sampleRate},
                             {"rate_family", family},
                             {"n_taps", taps},
                             {"n_channels", 4},
                             {"channel_order", {"LL", "LR", "RL", "RR"}},
                             {"phase_type", "original"},
                             {"normalization", "ild_preserving"},
                             {"max_dc_gain", 1.0},
                             {"source_azimuth_left", -30.0},
                             {"source_azimuth_right", 30.0},
                             {"source_elevation", 0.0},
                             {"license", "Test"},
                             {"attribution", "Test Data"},
                             {"source", "synthetic"},
                             {"storage_format", "channel_major_v1"}};

                std::string jsonPath = testDir_ + "/hrtf_" + size + "_" + family + ".json";
                std::ofstream jsonFile(jsonPath);
                jsonFile << meta.dump(2);
                jsonFile.close();
            }
        }
    }

    std::string testDir_;
};

// Test: HRTFProcessor initialization
TEST_F(HRTFProcessorTest, Initialize) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    EXPECT_TRUE(processor.initialize(testDir_, 256, HeadSize::M, RateFamily::RATE_44K));
    EXPECT_TRUE(processor.isEnabled());
    EXPECT_EQ(processor.getCurrentHeadSize(), HeadSize::M);
    EXPECT_EQ(processor.getCurrentRateFamily(), RateFamily::RATE_44K);
}

// Test: HRTFProcessor fails with invalid directory
TEST_F(HRTFProcessorTest, InitializeInvalidDir) {
    HRTFProcessor processor;
    EXPECT_FALSE(processor.initialize("/nonexistent/path"));
}

// Test: Initialize succeeds when only 44k family exists
TEST_F(HRTFProcessorTest, InitializeWithSingleRateFamily) {
    createTestHRTFFiles();

    // Remove 48k pair to simulate partial dataset
    std::filesystem::remove(testDir_ + "/hrtf_m_48k.bin");
    std::filesystem::remove(testDir_ + "/hrtf_m_48k.json");

    HRTFProcessor processor;
    EXPECT_TRUE(processor.initialize(testDir_, 256, HeadSize::M, RateFamily::RATE_44K));
}

// Test: Reject mismatched tap counts between rate families
TEST_F(HRTFProcessorTest, RejectsMismatchedTapCountsAcrossRates) {
    // 44k uses 256 taps, 48k uses 128 taps -> should be rejected
    createTestHRTFFiles(256, 128);

    HRTFProcessor processor;
    EXPECT_FALSE(processor.initialize(testDir_, 256, HeadSize::M, RateFamily::RATE_44K));
}

// Test: Enable/disable crossfeed
TEST_F(HRTFProcessorTest, EnableDisable) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_));

    EXPECT_TRUE(processor.isEnabled());
    processor.setEnabled(false);
    EXPECT_FALSE(processor.isEnabled());
    processor.setEnabled(true);
    EXPECT_TRUE(processor.isEnabled());
}

// Test: Passthrough when disabled
TEST_F(HRTFProcessorTest, PassthroughWhenDisabled) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_));

    processor.setEnabled(false);

    // Create test input
    std::vector<float> inputL(1024, 1.0f);
    std::vector<float> inputR(1024, 2.0f);
    std::vector<float> outputL, outputR;

    ASSERT_TRUE(
        processor.processStereo(inputL.data(), inputR.data(), inputL.size(), outputL, outputR));

    // Output should equal input (passthrough)
    ASSERT_EQ(outputL.size(), inputL.size());
    ASSERT_EQ(outputR.size(), inputR.size());

    for (size_t i = 0; i < inputL.size(); ++i) {
        EXPECT_FLOAT_EQ(outputL[i], inputL[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(outputR[i], inputR[i]) << "Mismatch at index " << i;
    }
}

// Test: Basic crossfeed processing
TEST_F(HRTFProcessorTest, BasicProcessing) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    // Create impulse in left channel only
    std::vector<float> inputL(1024, 0.0f);
    std::vector<float> inputR(1024, 0.0f);
    inputL[0] = 1.0f;  // Impulse

    std::vector<float> outputL, outputR;

    ASSERT_TRUE(
        processor.processStereo(inputL.data(), inputR.data(), inputL.size(), outputL, outputR));

    ASSERT_EQ(outputL.size(), inputL.size());
    ASSERT_EQ(outputR.size(), inputR.size());

    // Left output should have impulse at t=0 (LL path)
    // Right output should have weaker impulse with ITD delay (LR path)
    EXPECT_GT(std::abs(outputL[0]), 0.5f) << "Left output should have strong impulse";

    // Find peak in right channel (should be delayed)
    float maxR = 0.0f;
    int maxRIdx = 0;
    for (size_t i = 0; i < outputR.size(); ++i) {
        if (std::abs(outputR[i]) > maxR) {
            maxR = std::abs(outputR[i]);
            maxRIdx = i;
        }
    }
    EXPECT_GT(maxRIdx, 10) << "Right output peak should be delayed (ITD)";
    EXPECT_LT(maxR, std::abs(outputL[0]))
        << "Contralateral (LR) should be weaker than ipsilateral (LL)";
}

TEST_F(HRTFProcessorTest, GenerateWoodworthProfileSetsCombinedFilter) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256, HeadSize::M, RateFamily::RATE_44K));
    ASSERT_FALSE(processor.isUsingCombinedFilter());

    HRTF::WoodworthParams params;
    EXPECT_TRUE(processor.generateWoodworthProfile(RateFamily::RATE_44K, 30.0f, params));
    EXPECT_TRUE(processor.isUsingCombinedFilter());
}

TEST_F(HRTFProcessorTest, WoodworthProfileMatchesUpsampledDomain) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256, HeadSize::M, RateFamily::RATE_44K));

    auto findPeakIndex = [](const std::vector<float>& data) {
        size_t idx = 0;
        float maxVal = 0.0f;
        for (size_t i = 0; i < data.size(); ++i) {
            float val = std::fabs(data[i]);
            if (val > maxVal) {
                maxVal = val;
                idx = i;
            }
        }
        return idx;
    };

    auto verifyFamily = [&](RateFamily family, float targetSampleRate) {
        if (processor.getCurrentRateFamily() != family) {
            ASSERT_TRUE(processor.switchRateFamily(family));
        }

        HRTF::WoodworthParams params;
        constexpr float azimuthDeg = 30.0f;
        ASSERT_TRUE(processor.generateWoodworthProfile(family, azimuthDeg, params));

        std::vector<float> inputL(512, 0.0f);
        std::vector<float> inputR(512, 0.0f);
        inputL[0] = 1.0f;

        std::vector<float> outputL;
        std::vector<float> outputR;
        ASSERT_TRUE(
            processor.processStereo(inputL.data(), inputR.data(), inputL.size(), outputL, outputR));

        size_t nearPeak = findPeakIndex(outputL);
        size_t farPeak = findPeakIndex(outputR);

        constexpr float kSpeedOfSound = 343.0f;
        float azRad = std::fabs(azimuthDeg) * static_cast<float>(M_PI) / 180.0f;
        float itdNear = (params.headRadiusMeters / kSpeedOfSound) * std::sin(azRad);
        float itdFar = (params.headRadiusMeters / kSpeedOfSound) * (azRad + std::sin(azRad));

        float expectedNearSamples = itdNear * targetSampleRate;
        float expectedFarSamples = itdFar * targetSampleRate;

        EXPECT_NEAR(static_cast<float>(nearPeak), expectedNearSamples, 6.0f)
            << "Near ear ITD mismatch for rate family " << rateFamilyToString(family);
        EXPECT_NEAR(static_cast<float>(farPeak), expectedFarSamples, 6.0f)
            << "Far ear ITD mismatch for rate family " << rateFamilyToString(family);
    };

    verifyFamily(RateFamily::RATE_44K, 705600.0f);
    verifyFamily(RateFamily::RATE_48K, 768000.0f);
}

// Test: Stereo symmetry
TEST_F(HRTFProcessorTest, StereoSymmetry) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    // Impulse in right channel
    std::vector<float> inputL(1024, 0.0f);
    std::vector<float> inputR(1024, 0.0f);
    inputR[0] = 1.0f;

    std::vector<float> outputL, outputR;
    ASSERT_TRUE(
        processor.processStereo(inputL.data(), inputR.data(), inputL.size(), outputL, outputR));

    // Right output should have impulse (RR path)
    EXPECT_GT(std::abs(outputR[0]), 0.5f);

    // Left should have delayed crossfeed (RL path)
    float maxL = 0.0f;
    int maxLIdx = 0;
    for (size_t i = 0; i < outputL.size(); ++i) {
        if (std::abs(outputL[i]) > maxL) {
            maxL = std::abs(outputL[i]);
            maxLIdx = i;
        }
    }
    EXPECT_GT(maxLIdx, 10) << "Left output peak should be delayed (ITD)";
}

// Test: Energy conservation (output energy <= input energy)
TEST_F(HRTFProcessorTest, EnergyConservation) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    // Create test signal
    std::vector<float> inputL(4096);
    std::vector<float> inputR(4096);
    for (size_t i = 0; i < inputL.size(); ++i) {
        inputL[i] = std::sin(2.0 * M_PI * 1000.0 * i / 705600.0);
        inputR[i] = std::cos(2.0 * M_PI * 1000.0 * i / 705600.0);
    }

    std::vector<float> outputL, outputR;
    ASSERT_TRUE(
        processor.processStereo(inputL.data(), inputR.data(), inputL.size(), outputL, outputR));

    // Calculate energies
    double inputEnergy = 0.0, outputEnergy = 0.0;
    for (size_t i = 0; i < inputL.size(); ++i) {
        inputEnergy += inputL[i] * inputL[i] + inputR[i] * inputR[i];
    }
    for (size_t i = 0; i < outputL.size(); ++i) {
        outputEnergy += outputL[i] * outputL[i] + outputR[i] * outputR[i];
    }

    // Allow some tolerance for filter effects
    EXPECT_LT(outputEnergy, inputEnergy * 2.0) << "Output energy should not be excessive";
    EXPECT_GT(outputEnergy, inputEnergy * 0.1) << "Output energy should not be too low";
}

// Test: Streaming mode initialization
TEST_F(HRTFProcessorTest, StreamingInit) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    EXPECT_TRUE(processor.initializeStreaming());
    EXPECT_GT(processor.getStreamValidInputPerBlock(), 0u);
}

// Test: Reset streaming
TEST_F(HRTFProcessorTest, ResetStreaming) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));
    ASSERT_TRUE(processor.initializeStreaming());

    // Process some data
    std::vector<float> inputL(1024, 1.0f);
    std::vector<float> inputR(1024, 1.0f);
    std::vector<float> outputL, outputR;
    processor.processStereo(inputL.data(), inputR.data(), inputL.size(), outputL, outputR);

    // Reset should clear overlap buffers
    processor.resetStreaming();

    // After reset, processing should give same results as fresh start
    // (This is a basic test - actual verification would need more infrastructure)
}

// Test: Metadata access
TEST_F(HRTFProcessorTest, MetadataAccess) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_));

    const auto& meta = processor.getCurrentMetadata();
    EXPECT_EQ(meta.nChannels, 4);
    EXPECT_EQ(meta.phaseType, "original");
    EXPECT_EQ(meta.normalization, "ild_preserving");
}

// Test: setCombinedFilter API
TEST_F(HRTFProcessorTest, SetCombinedFilter) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    // Get expected filter size
    size_t filterFftSize = processor.getFilterFftSize();
    ASSERT_GT(filterFftSize, 0u);

    // Create synthetic combined filter data (FFT'd complex values)
    std::vector<cufftComplex> combinedLL(filterFftSize);
    std::vector<cufftComplex> combinedLR(filterFftSize);
    std::vector<cufftComplex> combinedRL(filterFftSize);
    std::vector<cufftComplex> combinedRR(filterFftSize);

    // Initialize with identity-like response (DC gain = 1)
    for (size_t i = 0; i < filterFftSize; ++i) {
        combinedLL[i] = make_cuFloatComplex(1.0f, 0.0f);
        combinedLR[i] = make_cuFloatComplex(0.1f, 0.0f);  // Some crossfeed
        combinedRL[i] = make_cuFloatComplex(0.1f, 0.0f);
        combinedRR[i] = make_cuFloatComplex(1.0f, 0.0f);
    }

    // Apply combined filter for 44k family
    EXPECT_FALSE(processor.isUsingCombinedFilter());
    ASSERT_TRUE(processor.setCombinedFilter(RateFamily::RATE_44K, combinedLL.data(),
                                            combinedLR.data(), combinedRL.data(), combinedRR.data(),
                                            filterFftSize));
    EXPECT_TRUE(processor.isUsingCombinedFilter());

    // Wrong size should fail
    std::vector<cufftComplex> wrongSize(filterFftSize + 10);
    EXPECT_FALSE(processor.setCombinedFilter(RateFamily::RATE_44K, wrongSize.data(),
                                             wrongSize.data(), wrongSize.data(), wrongSize.data(),
                                             wrongSize.size()));

    // Clear combined filter
    processor.clearCombinedFilter();
    EXPECT_FALSE(processor.isUsingCombinedFilter());
}

// Test: setCombinedFilter with rate family switching
TEST_F(HRTFProcessorTest, SetCombinedFilterRateFamilySwitch) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    size_t filterFftSize = processor.getFilterFftSize();
    std::vector<cufftComplex> filter44k(filterFftSize);
    std::vector<cufftComplex> filter48k(filterFftSize);

    // Initialize filters with different values to distinguish them
    for (size_t i = 0; i < filterFftSize; ++i) {
        filter44k[i] = make_cuFloatComplex(1.0f, 0.0f);
        filter48k[i] = make_cuFloatComplex(2.0f, 0.0f);
    }

    // Set both rate families
    ASSERT_TRUE(processor.setCombinedFilter(RateFamily::RATE_44K, filter44k.data(),
                                            filter44k.data(), filter44k.data(), filter44k.data(),
                                            filterFftSize));
    ASSERT_TRUE(processor.setCombinedFilter(RateFamily::RATE_48K, filter48k.data(),
                                            filter48k.data(), filter48k.data(), filter48k.data(),
                                            filterFftSize));

    // Switch between rate families should maintain combined filter mode
    EXPECT_TRUE(processor.isUsingCombinedFilter());
    ASSERT_TRUE(processor.switchRateFamily(RateFamily::RATE_48K));
    EXPECT_TRUE(processor.isUsingCombinedFilter());
    ASSERT_TRUE(processor.switchRateFamily(RateFamily::RATE_44K));
    EXPECT_TRUE(processor.isUsingCombinedFilter());
}

// Test: Combined filter auto-restore after fallback to predefined
// Regression test for: 44k combined only → 48k (predefined) → 44k (should restore combined)
TEST_F(HRTFProcessorTest, SetCombinedFilterAutoRestore) {
    createTestHRTFFiles();

    HRTFProcessor processor;
    ASSERT_TRUE(processor.initialize(testDir_, 256));

    size_t filterFftSize = processor.getFilterFftSize();
    std::vector<cufftComplex> filter44k(filterFftSize);

    // Initialize filter for 44k only
    for (size_t i = 0; i < filterFftSize; ++i) {
        filter44k[i] = make_cuFloatComplex(1.0f, 0.0f);
    }

    // Set combined filter for 44k ONLY (not 48k)
    ASSERT_TRUE(processor.setCombinedFilter(RateFamily::RATE_44K, filter44k.data(),
                                            filter44k.data(), filter44k.data(), filter44k.data(),
                                            filterFftSize));
    EXPECT_TRUE(processor.isUsingCombinedFilter());
    EXPECT_EQ(processor.getCurrentRateFamily(), RateFamily::RATE_44K);

    // Switch to 48k - should fall back to predefined (no combined filter for 48k)
    ASSERT_TRUE(processor.switchRateFamily(RateFamily::RATE_48K));
    EXPECT_FALSE(processor.isUsingCombinedFilter()) << "Should fall back to predefined for 48k";
    EXPECT_EQ(processor.getCurrentRateFamily(), RateFamily::RATE_48K);

    // Switch back to 44k - should AUTO-RESTORE combined filter
    ASSERT_TRUE(processor.switchRateFamily(RateFamily::RATE_44K));
    EXPECT_TRUE(processor.isUsingCombinedFilter())
        << "Should auto-restore combined filter when returning to 44k";
    EXPECT_EQ(processor.getCurrentRateFamily(), RateFamily::RATE_44K);
}

// Test: Helper functions
TEST(CrossfeedHelperTest, HeadSizeToString) {
    EXPECT_STREQ(headSizeToString(HeadSize::XS), "xs");
    EXPECT_STREQ(headSizeToString(HeadSize::S), "s");
    EXPECT_STREQ(headSizeToString(HeadSize::M), "m");
    EXPECT_STREQ(headSizeToString(HeadSize::L), "l");
    EXPECT_STREQ(headSizeToString(HeadSize::XL), "xl");
}

TEST(CrossfeedHelperTest, RateFamilyToString) {
    EXPECT_STREQ(rateFamilyToString(RateFamily::RATE_44K), "44k");
    EXPECT_STREQ(rateFamilyToString(RateFamily::RATE_48K), "48k");
}
