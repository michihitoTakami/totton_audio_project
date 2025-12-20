#include "audio/filter_headroom.h"
#include "gtest/gtest.h"

#include <cctype>
#include <filesystem>
#include <fstream>
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

    fs::path writeMetadata(const fs::path& coeffPath, float maxCoeff, float l1Norm) {
        fs::path metaPath = coeffPath;
        metaPath.replace_extension(".json");
        std::ofstream ofs(metaPath);
        ofs << R"({"validation_results":{"normalization":{)"
            << R"("max_coefficient_amplitude":)" << maxCoeff << R"(,)"
            << R"("l1_norm":)" << l1Norm << R"(}}})";
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
