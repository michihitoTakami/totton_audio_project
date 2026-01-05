#include "audio/filter_headroom.h"
#include "gtest/gtest.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

class FilterHeadroomCacheTest : public ::testing::Test {
   protected:
    fs::path tempDir;

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = "filter_headroom";
        if (info) {
            name = std::string(info->test_suite_name()) + "_" + std::string(info->name());
        }
        for (char& c : name) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
                c = '_';
            }
        }
        tempDir = fs::temp_directory_path() /
                  ("gpu_upsampler_test_" + name + "_" + std::to_string(getpid()));
        fs::create_directories(tempDir);
    }

    void TearDown() override {
        fs::remove_all(tempDir);
    }

    fs::path writeMetadata(const fs::path& coeffPath, float maxCoeff, float l1Norm,
                           int inputRate = 44100) {
        fs::path metaPath = coeffPath;
        metaPath.replace_extension(".json");
        std::ofstream ofs(metaPath);
        ofs << R"({"sample_rate_input":)" << inputRate << R"(,)"
            << R"("validation_results":{"normalization":{)"
            << R"("max_coefficient_amplitude":)" << maxCoeff << R"(,)"
            << R"("l1_norm":)" << l1Norm << R"(}}})";
        return metaPath;
    }

    fs::path writeMetadataWithDcGain(const fs::path& coeffPath, float maxCoeff, float l1Norm,
                                     float upsampleRatio, float normalizedDcGain,
                                     std::optional<float> inputPeakNormalized = std::nullopt,
                                     int inputRate = 44100) {
        fs::path metaPath = coeffPath;
        metaPath.replace_extension(".json");
        std::ofstream ofs(metaPath);
        int outputRate = static_cast<int>(inputRate * upsampleRatio);
        ofs << R"({"sample_rate_input":)" << inputRate << R"(,"sample_rate_output":)" << outputRate
            << R"(,"upsample_ratio":)" << upsampleRatio << R"(,)"
            << R"("validation_results":{"normalization":{)"
            << R"("max_coefficient_amplitude":)" << maxCoeff << R"(,)"
            << R"("l1_norm":)" << l1Norm << R"(,)"
            << R"("normalized_dc_gain":)" << normalizedDcGain << R"(})";
        if (inputPeakNormalized.has_value()) {
            ofs << R"(,"input_band_peak_normalized":)" << inputPeakNormalized.value();
        }
        ofs << "}}";
        return metaPath;
    }
};

TEST_F(FilterHeadroomCacheTest, MissingMetadataUsesDefaults) {
    FilterHeadroomCache cache(0.8f);
    fs::path coeffPath = tempDir / "filter.bin";

    FilterHeadroomInfo info = cache.get(coeffPath.string());
    EXPECT_FALSE(info.metadataFound);
    EXPECT_FLOAT_EQ(info.maxCoefficient, 1.0f);
    EXPECT_FLOAT_EQ(info.safeGain, 0.8f);
    EXPECT_FLOAT_EQ(info.targetPeak, 0.8f);
    EXPECT_EQ(info.metadataPath, (tempDir / "filter.json").string());
}

TEST_F(FilterHeadroomCacheTest, LoadsMetadataAndComputesSafeGain) {
    FilterHeadroomCache cache(0.9f);
    fs::path coeffPath = tempDir / "filter.bin";
    writeMetadata(coeffPath, 2.0f, 123.0f);

    FilterHeadroomInfo info = cache.get(coeffPath.string());
    EXPECT_TRUE(info.metadataFound);
    EXPECT_FLOAT_EQ(info.maxCoefficient, 2.0f);
    EXPECT_FLOAT_EQ(info.l1Norm, 123.0f);
    EXPECT_FLOAT_EQ(info.safeGain, 0.45f);
}

TEST_F(FilterHeadroomCacheTest, TargetPeakChangeClearsCache) {
    FilterHeadroomCache cache(0.9f);
    fs::path coeffPath = tempDir / "filter.bin";
    writeMetadata(coeffPath, 2.0f, 0.0f);

    FilterHeadroomInfo first = cache.get(coeffPath.string());
    EXPECT_FLOAT_EQ(first.safeGain, 0.45f);

    cache.setTargetPeak(0.7f);
    FilterHeadroomInfo second = cache.get(coeffPath.string());
    EXPECT_FLOAT_EQ(second.safeGain, 0.35f);
}

TEST_F(FilterHeadroomCacheTest, PrefersInputPeakWhenAvailable) {
    FilterHeadroomCache cache(0.92f);
    fs::path coeffPath = tempDir / "filter_input_peak.bin";
    writeMetadataWithDcGain(coeffPath, 2.0f, 0.0f, 16.0f, 15.84f, 0.99f);

    FilterHeadroomInfo info = cache.get(coeffPath.string());
    EXPECT_TRUE(info.metadataFound);
    EXPECT_TRUE(info.usedInputBandPeak);
    EXPECT_FLOAT_EQ(info.inputBandPeak, 0.99f);
    EXPECT_NEAR(info.safeGain, 0.92f / 0.99f, 1e-6f);
}

TEST_F(FilterHeadroomCacheTest, DerivesInputPeakFromDcGainWhenMissing) {
    FilterHeadroomCache cache(0.92f);
    fs::path coeffPath = tempDir / "filter_derived_peak.bin";
    writeMetadataWithDcGain(coeffPath, 2.0f, 0.0f, 8.0f, 7.92f);

    FilterHeadroomInfo info = cache.get(coeffPath.string());
    EXPECT_TRUE(info.metadataFound);
    EXPECT_TRUE(info.usedInputBandPeak);
    EXPECT_NEAR(info.inputBandPeak, 0.99f, 1e-6f);
    EXPECT_NEAR(info.safeGain, 0.92f / 0.99f, 1e-6f);
}

TEST_F(FilterHeadroomCacheTest, FamilyModeUsesMaxAcrossPaths) {
    FilterHeadroomCache cache(0.9f);
    cache.setMode(HeadroomMode::FamilyMax);
    fs::path coeffPathA = tempDir / "filter_a.bin";
    fs::path coeffPathB = tempDir / "filter_b.bin";
    writeMetadata(coeffPathA, 1.5f, 0.0f);
    writeMetadata(coeffPathB, 2.0f, 0.0f);

    FilterHeadroomInfo first = cache.get(coeffPathA.string());
    EXPECT_TRUE(first.metadataFound);
    EXPECT_NEAR(first.safeGain, 0.9f / 1.5f, 1e-6f);

    FilterHeadroomInfo second = cache.get(coeffPathB.string());
    EXPECT_TRUE(second.metadataFound);
    EXPECT_NEAR(second.safeGain, 0.9f / 2.0f, 1e-6f);

    FilterHeadroomInfo updatedFirst = cache.get(coeffPathA.string());
    EXPECT_NEAR(updatedFirst.safeGain, 0.9f / 2.0f, 1e-6f);
    EXPECT_TRUE(updatedFirst.familyGainApplied);
}
