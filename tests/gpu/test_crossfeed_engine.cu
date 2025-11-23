#include <gtest/gtest.h>
#include "crossfeed_engine.h"
#include <cmath>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace CrossfeedEngine;

// Test fixture for HRTFProcessor tests
class HRTFProcessorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create test data directory
        testDir_ = "test_hrtf_data";
        system(("mkdir -p " + testDir_).c_str());
    }

    void TearDown() override {
        // Clean up test data
        system(("rm -rf " + testDir_).c_str());
    }

    // Create synthetic HRTF filter coefficients for testing
    // Simple impulse response that approximates crossfeed
    void createTestHRTFFiles(int taps = 256) {
        // Create 4 channels: LL, LR, RL, RR
        // LL/RR: Strong (ipsilateral) - direct path
        // LR/RL: Weaker (contralateral) - cross path with delay
        std::vector<float> ll(taps, 0.0f);
        std::vector<float> lr(taps, 0.0f);
        std::vector<float> rl(taps, 0.0f);
        std::vector<float> rr(taps, 0.0f);

        // LL: Direct path impulse at t=0
        ll[0] = 1.0f;

        // LR: Contralateral with ITD delay (~20 samples at 705.6kHz ~ 28us)
        // and ILD attenuation (0.5)
        int itdSamples = 20;
        lr[itdSamples] = 0.5f;

        // RL: Same as LR (symmetric head)
        rl[itdSamples] = 0.5f;

        // RR: Same as LL
        rr[0] = 1.0f;

        // Write binary file (4 channels interleaved)
        for (const char* size : {"m"}) {
            for (const char* family : {"44k", "48k"}) {
                std::string binPath = testDir_ + "/hrtf_" + size + "_" + family + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                binFile.write(reinterpret_cast<const char*>(ll.data()), taps * sizeof(float));
                binFile.write(reinterpret_cast<const char*>(lr.data()), taps * sizeof(float));
                binFile.write(reinterpret_cast<const char*>(rl.data()), taps * sizeof(float));
                binFile.write(reinterpret_cast<const char*>(rr.data()), taps * sizeof(float));
                binFile.close();

                // Write JSON metadata
                int sampleRate = (strcmp(family, "44k") == 0) ? 705600 : 768000;
                json meta = {
                    {"description", "Test HRTF"},
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
                    {"source", "synthetic"}
                };

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

    ASSERT_TRUE(processor.processStereo(inputL.data(), inputR.data(), inputL.size(),
                                         outputL, outputR));

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

    ASSERT_TRUE(processor.processStereo(inputL.data(), inputR.data(), inputL.size(),
                                         outputL, outputR));

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
    EXPECT_LT(maxR, std::abs(outputL[0])) << "Contralateral (LR) should be weaker than ipsilateral (LL)";
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
    ASSERT_TRUE(processor.processStereo(inputL.data(), inputR.data(), inputL.size(),
                                         outputL, outputR));

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
    ASSERT_TRUE(processor.processStereo(inputL.data(), inputR.data(), inputL.size(),
                                         outputL, outputR));

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

// Test: Helper functions
TEST(CrossfeedHelperTest, HeadSizeToString) {
    EXPECT_STREQ(headSizeToString(HeadSize::S), "s");
    EXPECT_STREQ(headSizeToString(HeadSize::M), "m");
    EXPECT_STREQ(headSizeToString(HeadSize::L), "l");
    EXPECT_STREQ(headSizeToString(HeadSize::XL), "xl");
}

TEST(CrossfeedHelperTest, RateFamilyToString) {
    EXPECT_STREQ(rateFamilyToString(RateFamily::RATE_44K), "44k");
    EXPECT_STREQ(rateFamilyToString(RateFamily::RATE_48K), "48k");
}
