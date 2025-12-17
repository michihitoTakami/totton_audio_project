#include "convolution_engine.h"

#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

using json = nlohmann::json;
using namespace ConvolutionEngine;

namespace {

std::vector<std::string> allHeadSizes() {
    return {"xs", "s", "m", "l", "xl"};
}

void writeHrtfFiles(const std::filesystem::path& dir, int taps,
                    const std::array<float, 4>& gains = {1.0f, 0.5f, 0.25f, 0.75f}) {
    std::filesystem::create_directories(dir);
    for (const auto& size : allHeadSizes()) {
        for (const auto& family : {"44k", "48k"}) {
            std::vector<float> coeffs(static_cast<size_t>(taps) * 4, 0.0f);
            // channel-major: LL, LR, RL, RR
            coeffs[0] = gains[0];
            coeffs[taps] = gains[1];
            coeffs[taps * 2] = gains[2];
            coeffs[taps * 3] = gains[3];

            const auto binPath = dir / ("hrtf_" + size + std::string("_") + family + ".bin");
            std::ofstream binFile(binPath, std::ios::binary);
            binFile.write(reinterpret_cast<const char*>(coeffs.data()),
                          coeffs.size() * sizeof(float));

            json meta = {{"description", "test hrtf"},
                         {"size_category", size},
                         {"sample_rate", std::string(family) == "44k" ? 705600 : 768000},
                         {"rate_family", family},
                         {"n_taps", taps},
                         {"n_channels", 4},
                         {"channel_order", {"LL", "LR", "RL", "RR"}},
                         {"storage_format", "channel_major_v1"}};
            const auto jsonPath = dir / ("hrtf_" + size + std::string("_") + family + ".json");
            std::ofstream jsonFile(jsonPath);
            jsonFile << meta.dump(2);
        }
    }
}

size_t expectedValid(int blockSize, int taps) {
    size_t need = static_cast<size_t>(blockSize) + static_cast<size_t>(taps) - 1;
    size_t fft = 1;
    while (fft < need) {
        fft <<= 1;
    }
    return fft - static_cast<size_t>(taps - 1);
}

}  // namespace

class FourChannelFIRTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = info ? std::string(info->test_suite_name()) + "_" + info->name()
                                : "four_channel_fir";
        for (char& c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        tempDir_ = std::filesystem::temp_directory_path() /
                   ("four_channel_fir_" + name + "_" + std::to_string(getpid()));
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(tempDir_, ec);
    }

    std::filesystem::path tempDir_;
};

TEST_F(FourChannelFIRTest, InitializeWithFiveHeadSizes) {
    const int taps = 4;
    writeHrtfFiles(tempDir_, taps);

    FourChannelFIR fir;
    ASSERT_TRUE(fir.initialize(tempDir_.string(), 8, HeadSize::M, RateFamily::RATE_44K));
    EXPECT_EQ(fir.getFilterTaps(), taps);
    EXPECT_EQ(fir.getStreamValidInputPerBlock(), expectedValid(8, taps));
    EXPECT_TRUE(fir.switchHeadSize(HeadSize::XL));
    EXPECT_EQ(fir.getCurrentHeadSize(), HeadSize::XL);
}

TEST_F(FourChannelFIRTest, ProcessImpulseCombinesChannels) {
    const int taps = 1;
    writeHrtfFiles(tempDir_, taps, {1.0f, 0.5f, 0.25f, 0.75f});

    FourChannelFIR fir;
    ASSERT_TRUE(fir.initialize(tempDir_.string(), 8, HeadSize::M, RateFamily::RATE_44K));
    ASSERT_TRUE(fir.initializeStreaming());

    const size_t block = fir.getStreamValidInputPerBlock();
    StreamFloatVector streamInputL(block, 0.0f);
    StreamFloatVector streamInputR(block, 0.0f);
    size_t accumulatedL = 0;
    size_t accumulatedR = 0;

    StreamFloatVector outputL(fir.getValidOutputPerBlock(), 0.0f);
    StreamFloatVector outputR(fir.getValidOutputPerBlock(), 0.0f);

    std::vector<float> inputL(block, 0.0f);
    std::vector<float> inputR(block, 0.0f);
    inputL[0] = 1.0f;  // impulse on left

    ASSERT_TRUE(fir.processStreamBlock(inputL.data(), inputR.data(), block, outputL, outputR,
                                       nullptr, streamInputL, streamInputR, accumulatedL,
                                       accumulatedR));

    ASSERT_EQ(outputL.size(), fir.getValidOutputPerBlock());
    ASSERT_EQ(outputR.size(), fir.getValidOutputPerBlock());
    EXPECT_FLOAT_EQ(outputL[0], 1.0f);
    EXPECT_FLOAT_EQ(outputR[0], 0.5f);
    for (size_t i = 1; i < outputL.size(); ++i) {
        EXPECT_FLOAT_EQ(outputL[i], 0.0f);
        EXPECT_FLOAT_EQ(outputR[i], 0.0f);
    }
    EXPECT_EQ(accumulatedL, 0U);
    EXPECT_EQ(accumulatedR, 0U);
}
