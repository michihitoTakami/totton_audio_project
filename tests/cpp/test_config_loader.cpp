/**
 * @file test_config_loader.cpp
 * @brief Unit tests for config loader (JSON configuration)
 */

#include "core/config_loader.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace fs = std::filesystem;

class ConfigLoaderTest : public ::testing::Test {
   protected:
    fs::path tempDir;
    fs::path testConfigPath;

    void SetUp() override {
        // Create temp directory for test files
        tempDir = fs::temp_directory_path() / "gpu_upsampler_test";
        fs::create_directories(tempDir);
        testConfigPath = tempDir / "test_config.json";
    }

    void TearDown() override {
        // Clean up temp files
        fs::remove_all(tempDir);
    }

    void writeConfig(const std::string& content) {
        std::ofstream file(testConfigPath);
        file << content;
        file.close();
    }
};

// ============================================================
// loadAppConfig tests
// ============================================================

TEST_F(ConfigLoaderTest, LoadNonExistentFileReturnsFalse) {
    AppConfig config;
    bool result = loadAppConfig("/nonexistent/path/config.json", config, false);

    EXPECT_FALSE(result);
}

TEST_F(ConfigLoaderTest, LoadNonExistentFileUsesDefaults) {
    AppConfig config;
    loadAppConfig("/nonexistent/path/config.json", config, false);

    // Should have default values
    EXPECT_EQ(config.alsaDevice, "hw:USB");
    EXPECT_EQ(config.bufferSize, 262144);
    EXPECT_EQ(config.periodSize, 32768);
    EXPECT_EQ(config.upsampleRatio, 16);
    EXPECT_EQ(config.blockSize, 4096);
    EXPECT_FLOAT_EQ(config.gain, 1.0f);
    EXPECT_FALSE(config.eqEnabled);
    EXPECT_EQ(config.eqProfilePath, "");
}

TEST_F(ConfigLoaderTest, LoadEmptyJsonReturnsTrue) {
    writeConfig("{}");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
}

TEST_F(ConfigLoaderTest, LoadFullConfig) {
    writeConfig(R"({
        "alsaDevice": "hw:Audio",
        "bufferSize": 131072,
        "periodSize": 16384,
        "upsampleRatio": 8,
        "blockSize": 8192,
        "gain": 8.0,
        "filterPath": "custom/filter.bin",
        "eqEnabled": true,
        "eqProfilePath": "data/EQ/custom.txt"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.alsaDevice, "hw:Audio");
    EXPECT_EQ(config.bufferSize, 131072);
    EXPECT_EQ(config.periodSize, 16384);
    EXPECT_EQ(config.upsampleRatio, 8);
    EXPECT_EQ(config.blockSize, 8192);
    EXPECT_FLOAT_EQ(config.gain, 8.0f);
    EXPECT_EQ(config.filterPath, "custom/filter.bin");
    EXPECT_TRUE(config.eqEnabled);
    EXPECT_EQ(config.eqProfilePath, "data/EQ/custom.txt");
    EXPECT_EQ(config.output.mode, OutputMode::Usb);
    EXPECT_EQ(config.output.usb.preferredDevice, "hw:Audio");
}

TEST_F(ConfigLoaderTest, LoadPartialConfigKeepsDefaults) {
    writeConfig(R"({
        "alsaDevice": "hw:Custom",
        "upsampleRatio": 4
    })");

    AppConfig config;
    loadAppConfig(testConfigPath, config, false);

    // Specified values
    EXPECT_EQ(config.alsaDevice, "hw:Custom");
    EXPECT_EQ(config.upsampleRatio, 4);

    // Default values for unspecified fields
    EXPECT_EQ(config.bufferSize, 262144);
    EXPECT_EQ(config.periodSize, 32768);
    EXPECT_EQ(config.blockSize, 4096);
    EXPECT_FLOAT_EQ(config.gain, 1.0f);
}

TEST_F(ConfigLoaderTest, LoadInvalidJsonReturnsFalse) {
    writeConfig("{ invalid json }");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_FALSE(result);
}

TEST_F(ConfigLoaderTest, LoadConfigWithExtraFieldsIgnoresThem) {
    writeConfig(R"({
        "alsaDevice": "hw:Test",
        "unknownField": "should be ignored",
        "anotherUnknown": 12345
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.alsaDevice, "hw:Test");
}

TEST_F(ConfigLoaderTest, LoadOutputSectionOverridesLegacyAlsaDevice) {
    writeConfig(R"({
        "alsaDevice": "hw:Legacy",
        "output": {
            "mode": "usb",
            "options": {
                "usb": {
                    "preferredDevice": "hw:Preferred"
                }
            }
        }
    })");

    AppConfig config;
    ASSERT_TRUE(loadAppConfig(testConfigPath, config, false));
    EXPECT_EQ(config.alsaDevice, "hw:Preferred");
    EXPECT_EQ(config.output.usb.preferredDevice, "hw:Preferred");
}

TEST_F(ConfigLoaderTest, UnsupportedOutputModeFallsBackToUsb) {
    writeConfig(R"({
        "alsaDevice": "hw:Audio",
        "output": {
            "mode": "spdif",
            "options": {
                "usb": {
                    "preferredDevice": "hw:Audio"
                }
            }
        }
    })");

    AppConfig config;
    ASSERT_TRUE(loadAppConfig(testConfigPath, config, false));
    EXPECT_EQ(config.output.mode, OutputMode::Usb);
}

// ============================================================
// AppConfig default values tests
// ============================================================

TEST_F(ConfigLoaderTest, AppConfigDefaultValues) {
    AppConfig config;

    EXPECT_EQ(config.alsaDevice, "hw:USB");
    EXPECT_EQ(config.bufferSize, 262144);
    EXPECT_EQ(config.periodSize, 32768);
    EXPECT_EQ(config.upsampleRatio, 16);
    EXPECT_EQ(config.blockSize, 4096);
    EXPECT_FLOAT_EQ(config.gain, 1.0f);
    EXPECT_EQ(config.filterPath, "data/coefficients/filter_44k_16x_2m_linear_phase.bin");
    EXPECT_EQ(config.phaseType, PhaseType::Minimum);
    EXPECT_FALSE(config.eqEnabled);
    EXPECT_EQ(config.eqProfilePath, "");
    EXPECT_EQ(config.output.mode, OutputMode::Usb);
    EXPECT_EQ(config.output.usb.preferredDevice, "hw:USB");
}

// ============================================================
// DEFAULT_CONFIG_FILE constant test
// ============================================================

TEST_F(ConfigLoaderTest, DefaultConfigFileConstant) {
    EXPECT_STREQ(DEFAULT_CONFIG_FILE, "config.json");
}

// ============================================================
// PhaseType tests
// ============================================================

TEST_F(ConfigLoaderTest, ParsePhaseTypeMinimum) {
    EXPECT_EQ(parsePhaseType("minimum"), PhaseType::Minimum);
}

TEST_F(ConfigLoaderTest, ParsePhaseTypeLinear) {
    EXPECT_EQ(parsePhaseType("linear"), PhaseType::Linear);
}

TEST_F(ConfigLoaderTest, ParsePhaseTypeHybridAlias) {
    EXPECT_EQ(parsePhaseType("hybrid"), PhaseType::Linear);
}

TEST_F(ConfigLoaderTest, ParsePhaseTypeInvalidDefaultsToMinimum) {
    EXPECT_EQ(parsePhaseType("invalid"), PhaseType::Minimum);
    EXPECT_EQ(parsePhaseType(""), PhaseType::Minimum);
    EXPECT_EQ(parsePhaseType("MINIMUM"), PhaseType::Minimum);  // case sensitive
}

TEST_F(ConfigLoaderTest, PhaseTypeToStringMinimum) {
    EXPECT_STREQ(phaseTypeToString(PhaseType::Minimum), "minimum");
}

TEST_F(ConfigLoaderTest, PhaseTypeToStringLinear) {
    EXPECT_STREQ(phaseTypeToString(PhaseType::Linear), "hybrid");
}

TEST_F(ConfigLoaderTest, AppConfigDefaultPhaseType) {
    AppConfig config;
    EXPECT_EQ(config.phaseType, PhaseType::Minimum);
}

// ============================================================
// OutputMode tests
// ============================================================

TEST(ConfigLoaderStandaloneTests, ParseOutputModeUsb) {
    EXPECT_EQ(parseOutputMode("usb"), OutputMode::Usb);
    EXPECT_EQ(parseOutputMode("USB"), OutputMode::Usb);
}

TEST(ConfigLoaderStandaloneTests, ParseOutputModeInvalidDefaultsToUsb) {
    EXPECT_EQ(parseOutputMode("spdif"), OutputMode::Usb);
    EXPECT_EQ(parseOutputMode(""), OutputMode::Usb);
}

TEST(ConfigLoaderStandaloneTests, OutputModeToStringUsb) {
    EXPECT_STREQ(outputModeToString(OutputMode::Usb), "usb");
}

TEST_F(ConfigLoaderTest, PartitionedConvolutionDefaults) {
    AppConfig config;
    EXPECT_FALSE(config.partitionedConvolution.enabled);
    EXPECT_EQ(config.partitionedConvolution.fastPartitionTaps, 32768);
    EXPECT_EQ(config.partitionedConvolution.minPartitionTaps, 32768);
    EXPECT_EQ(config.partitionedConvolution.maxPartitions, 4);
    EXPECT_EQ(config.partitionedConvolution.tailFftMultiple, 2);
}

TEST_F(ConfigLoaderTest, LoadPartitionedConvolutionSection) {
    writeConfig(R"({
        "partitionedConvolution": {
            "enabled": true,
            "fastPartitionTaps": 48000,
            "minPartitionTaps": 8000,
            "maxPartitions": 6,
            "tailFftMultiple": 6
        }
    })");

    AppConfig config;
    ASSERT_TRUE(loadAppConfig(testConfigPath, config, false));

    EXPECT_TRUE(config.partitionedConvolution.enabled);
    EXPECT_EQ(config.partitionedConvolution.fastPartitionTaps, 48000);
    EXPECT_EQ(config.partitionedConvolution.minPartitionTaps, 8000);
    EXPECT_EQ(config.partitionedConvolution.maxPartitions, 6);
    EXPECT_EQ(config.partitionedConvolution.tailFftMultiple, 6);
}

TEST_F(ConfigLoaderTest, PartitionedConvolutionInvalidValuesClamped) {
    writeConfig(R"({
        "partitionedConvolution": {
            "enabled": true,
            "fastPartitionTaps": 256,
            "minPartitionTaps": -10,
            "maxPartitions": 0,
            "tailFftMultiple": 0
        }
    })");

    AppConfig config;
    ASSERT_TRUE(loadAppConfig(testConfigPath, config, false));

    EXPECT_TRUE(config.partitionedConvolution.enabled);
    // fastPartitionTaps/minPartitionTaps are clamped to at least 1024
    EXPECT_EQ(config.partitionedConvolution.fastPartitionTaps, 1024);
    EXPECT_EQ(config.partitionedConvolution.minPartitionTaps, 1024);
    EXPECT_EQ(config.partitionedConvolution.maxPartitions, 1);
    EXPECT_EQ(config.partitionedConvolution.tailFftMultiple, 2);
}

TEST_F(ConfigLoaderTest, LoadConfigWithPhaseTypeMinimum) {
    writeConfig(R"({"phaseType": "minimum"})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.phaseType, PhaseType::Minimum);
}

TEST_F(ConfigLoaderTest, LoadConfigWithPhaseTypeLinear) {
    writeConfig(R"({"phaseType": "linear"})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.phaseType, PhaseType::Linear);
}

TEST_F(ConfigLoaderTest, LoadConfigWithPhaseTypeHybridAlias) {
    writeConfig(R"({"phaseType": "hybrid"})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.phaseType, PhaseType::Linear);
}

TEST_F(ConfigLoaderTest, LoadConfigWithInvalidPhaseTypeDefaultsToMinimum) {
    writeConfig(R"({"phaseType": "invalid"})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.phaseType, PhaseType::Minimum);
}

// ============================================================
// Crossfeed settings tests
// ============================================================

TEST_F(ConfigLoaderTest, CrossfeedDefaultValues) {
    AppConfig config;

    EXPECT_FALSE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "m");
    EXPECT_EQ(config.crossfeed.hrtfPath, "data/crossfeed/hrtf/");
}

TEST_F(ConfigLoaderTest, LoadConfigWithCrossfeedEnabled) {
    writeConfig(R"({
        "crossfeed": {
            "enabled": true,
            "headSize": "l",
            "hrtfPath": "custom/hrtf/"
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "l");
    EXPECT_EQ(config.crossfeed.hrtfPath, "custom/hrtf/");
}

TEST_F(ConfigLoaderTest, LoadConfigWithCrossfeedDisabled) {
    writeConfig(R"({
        "crossfeed": {
            "enabled": false,
            "headSize": "s"
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "s");
    // hrtfPath should keep default
    EXPECT_EQ(config.crossfeed.hrtfPath, "data/crossfeed/hrtf/");
}

TEST_F(ConfigLoaderTest, LoadConfigWithPartialCrossfeedKeepsDefaults) {
    writeConfig(R"({
        "crossfeed": {
            "enabled": true
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.crossfeed.enabled);
    // headSize and hrtfPath should keep defaults
    EXPECT_EQ(config.crossfeed.headSize, "m");
    EXPECT_EQ(config.crossfeed.hrtfPath, "data/crossfeed/hrtf/");
}

TEST_F(ConfigLoaderTest, LoadConfigWithoutCrossfeedKeepsDefaults) {
    writeConfig(R"({
        "alsaDevice": "hw:Test"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "m");
    EXPECT_EQ(config.crossfeed.hrtfPath, "data/crossfeed/hrtf/");
}

TEST_F(ConfigLoaderTest, LoadConfigWithAllHeadSizes) {
    // Test all valid headSize values
    const char* headSizes[] = {"s", "m", "l", "xl"};
    for (const char* size : headSizes) {
        std::string json = R"({"crossfeed": {"headSize": ")" + std::string(size) + R"("}})";
        writeConfig(json);

        AppConfig config;
        bool result = loadAppConfig(testConfigPath, config, false);

        EXPECT_TRUE(result) << "Failed for headSize: " << size;
        EXPECT_EQ(config.crossfeed.headSize, size) << "Wrong headSize for: " << size;
    }
}

TEST_F(ConfigLoaderTest, LoadConfigWithInvalidHeadSizeDefaultsToMedium) {
    // Test invalid headSize values - should fallback to "m"
    const char* invalidSizes[] = {"xs", "xxl", "medium", "large", "", "123"};
    for (const char* size : invalidSizes) {
        std::string json = R"({"crossfeed": {"headSize": ")" + std::string(size) + R"("}})";
        writeConfig(json);

        AppConfig config;
        bool result = loadAppConfig(testConfigPath, config, false);

        EXPECT_TRUE(result) << "Failed for invalid headSize: " << size;
        EXPECT_EQ(config.crossfeed.headSize, "m") << "Should default to 'm' for: " << size;
    }
}

TEST_F(ConfigLoaderTest, LoadConfigWithCrossfeedWrongTypes) {
    // Test when crossfeed is not an object (should be ignored, keep defaults)
    writeConfig(R"({"crossfeed": "not an object"})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "m");
    EXPECT_EQ(config.crossfeed.hrtfPath, "data/crossfeed/hrtf/");
}

TEST_F(ConfigLoaderTest, LoadConfigWithCrossfeedArrayType) {
    // Test when crossfeed is an array (should be ignored, keep defaults)
    writeConfig(R"({"crossfeed": [1, 2, 3]})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "m");
}

TEST_F(ConfigLoaderTest, LoadConfigWithCrossfeedFieldsWrongTypes) {
    // Test when crossfeed fields have wrong types (should keep defaults for those fields)
    writeConfig(R"({
        "crossfeed": {
            "enabled": "not a bool",
            "headSize": 123,
            "hrtfPath": true
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    // All fields should keep defaults due to type mismatch
    EXPECT_FALSE(config.crossfeed.enabled);
    EXPECT_EQ(config.crossfeed.headSize, "m");
    EXPECT_EQ(config.crossfeed.hrtfPath, "data/crossfeed/hrtf/");
}

// ============================================================
// Issue #219: Multi-Rate settings tests
// ============================================================

TEST_F(ConfigLoaderTest, Issue219_MultiRateDefaultValues) {
    AppConfig config;

    EXPECT_FALSE(config.multiRateEnabled);
    EXPECT_EQ(config.coefficientDir, "data/coefficients");
}

TEST_F(ConfigLoaderTest, Issue219_LoadConfigWithMultiRateEnabled) {
    writeConfig(R"({
        "multiRateEnabled": true,
        "coefficientDir": "custom/coefficients"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.multiRateEnabled);
    EXPECT_EQ(config.coefficientDir, "custom/coefficients");
}

TEST_F(ConfigLoaderTest, Issue219_LoadConfigWithMultiRateDisabled) {
    writeConfig(R"({
        "multiRateEnabled": false,
        "coefficientDir": "other/coefficients"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.multiRateEnabled);
    EXPECT_EQ(config.coefficientDir, "other/coefficients");
}

TEST_F(ConfigLoaderTest, Issue219_LoadConfigWithoutMultiRateKeepsDefaults) {
    writeConfig(R"({
        "alsaDevice": "hw:Test"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.multiRateEnabled);
    EXPECT_EQ(config.coefficientDir, "data/coefficients");
}

TEST_F(ConfigLoaderTest, Issue219_LoadConfigWithPartialMultiRateKeepsDefaults) {
    writeConfig(R"({
        "multiRateEnabled": true
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.multiRateEnabled);
    // coefficientDir should keep default
    EXPECT_EQ(config.coefficientDir, "data/coefficients");
}

TEST_F(ConfigLoaderTest, Issue219_LoadConfigWithMultiRateAndFilterPaths) {
    writeConfig(R"({
        "multiRateEnabled": true,
        "filterPath44kMin": "a/path44kMin.bin",
        "filterPath48kLinear": "a/path48kLinear.bin",
        "coefficientDir": "multi/coefficients"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.multiRateEnabled);
    EXPECT_EQ(config.filterPath44kMin, "a/path44kMin.bin");
    EXPECT_EQ(config.filterPath48kLinear, "a/path48kLinear.bin");
    EXPECT_EQ(config.coefficientDir, "multi/coefficients");
}
