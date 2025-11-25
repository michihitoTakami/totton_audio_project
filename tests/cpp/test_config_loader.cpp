/**
 * @file test_config_loader.cpp
 * @brief Unit tests for config loader (JSON configuration)
 */

#include "config_loader.h"
#include "daemon_constants.h"

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
    EXPECT_FLOAT_EQ(config.headroomTarget, DaemonConstants::DEFAULT_HEADROOM_TARGET);
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
        "headroomTarget": 0.95,
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
    EXPECT_FLOAT_EQ(config.headroomTarget, 0.95f);
    EXPECT_EQ(config.filterPath, "custom/filter.bin");
    EXPECT_TRUE(config.eqEnabled);
    EXPECT_EQ(config.eqProfilePath, "data/EQ/custom.txt");
}

TEST_F(ConfigLoaderTest, HeadroomTargetClampedAboveMax) {
    writeConfig(R"({"headroomTarget": 5.0})");

    AppConfig config;
    loadAppConfig(testConfigPath, config, false);

    EXPECT_FLOAT_EQ(config.headroomTarget, DaemonConstants::MAX_HEADROOM_TARGET);
}

TEST_F(ConfigLoaderTest, HeadroomTargetClampedBelowMin) {
    writeConfig(R"({"headroomTarget": 0.1})");

    AppConfig config;
    loadAppConfig(testConfigPath, config, false);

    EXPECT_FLOAT_EQ(config.headroomTarget, DaemonConstants::MIN_HEADROOM_TARGET);
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
    EXPECT_FLOAT_EQ(config.headroomTarget, DaemonConstants::DEFAULT_HEADROOM_TARGET);
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
    EXPECT_EQ(config.filterPath, "data/coefficients/filter_44k_16x_2m_min_phase.bin");
    EXPECT_EQ(config.phaseType, PhaseType::Minimum);
    EXPECT_FALSE(config.eqEnabled);
    EXPECT_EQ(config.eqProfilePath, "");
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

TEST_F(ConfigLoaderTest, ParsePhaseTypeInvalidDefaultsToMinimum) {
    EXPECT_EQ(parsePhaseType("invalid"), PhaseType::Minimum);
    EXPECT_EQ(parsePhaseType(""), PhaseType::Minimum);
    EXPECT_EQ(parsePhaseType("MINIMUM"), PhaseType::Minimum);  // case sensitive
}

TEST_F(ConfigLoaderTest, PhaseTypeToStringMinimum) {
    EXPECT_STREQ(phaseTypeToString(PhaseType::Minimum), "minimum");
}

TEST_F(ConfigLoaderTest, PhaseTypeToStringLinear) {
    EXPECT_STREQ(phaseTypeToString(PhaseType::Linear), "linear");
}

TEST_F(ConfigLoaderTest, AppConfigDefaultPhaseType) {
    AppConfig config;
    EXPECT_EQ(config.phaseType, PhaseType::Minimum);
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

// ============================================================================
// Fallback Config Tests (Issue #139)
// ============================================================================

TEST_F(ConfigLoaderTest, FallbackDefaultValues) {
    AppConfig config;
    EXPECT_TRUE(config.fallback.enabled);
    EXPECT_FLOAT_EQ(config.fallback.gpuThreshold, 80.0f);
    EXPECT_EQ(config.fallback.gpuThresholdCount, 3);
    EXPECT_FLOAT_EQ(config.fallback.gpuRecoveryThreshold, 70.0f);
    EXPECT_EQ(config.fallback.gpuRecoveryCount, 5);
    EXPECT_TRUE(config.fallback.xrunTriggersFallback);
    EXPECT_EQ(config.fallback.monitorIntervalMs, 100);
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackEnabled) {
    writeConfig(R"({
        "fallback": {
            "enabled": true,
            "gpuThreshold": 90.0,
            "gpuThresholdCount": 5,
            "gpuRecoveryThreshold": 80.0,
            "gpuRecoveryCount": 10,
            "xrunTriggersFallback": false,
            "monitorIntervalMs": 200
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.fallback.enabled);
    EXPECT_FLOAT_EQ(config.fallback.gpuThreshold, 90.0f);
    EXPECT_EQ(config.fallback.gpuThresholdCount, 5);
    EXPECT_FLOAT_EQ(config.fallback.gpuRecoveryThreshold, 80.0f);
    EXPECT_EQ(config.fallback.gpuRecoveryCount, 10);
    EXPECT_FALSE(config.fallback.xrunTriggersFallback);
    EXPECT_EQ(config.fallback.monitorIntervalMs, 200);
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackDisabled) {
    writeConfig(R"({
        "fallback": {
            "enabled": false
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FALSE(config.fallback.enabled);
}

TEST_F(ConfigLoaderTest, LoadConfigWithoutFallbackKeepsDefaults) {
    writeConfig(R"({
        "alsaDevice": "hw:Test"
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.fallback.enabled);
    EXPECT_FLOAT_EQ(config.fallback.gpuThreshold, 80.0f);
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackValidation_GpuThresholdClamped) {
    // Test gpuThreshold clamping to 0-100%
    writeConfig(R"({
        "fallback": {
            "gpuThreshold": 150.0
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(config.fallback.gpuThreshold, 100.0f);  // Clamped to 100

    // Test negative value
    writeConfig(R"({
        "fallback": {
            "gpuThreshold": -10.0
        }
    })");
    result = loadAppConfig(testConfigPath, config, false);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(config.fallback.gpuThreshold, 0.0f);  // Clamped to 0
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackValidation_RecoveryThresholdClamped) {
    // Recovery threshold must be <= gpuThreshold
    writeConfig(R"({
        "fallback": {
            "gpuThreshold": 80.0,
            "gpuRecoveryThreshold": 90.0
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    // gpuRecoveryThreshold should be clamped to gpuThreshold
    EXPECT_FLOAT_EQ(config.fallback.gpuRecoveryThreshold, 80.0f);
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackValidation_CountValuesMin1) {
    writeConfig(R"({
        "fallback": {
            "gpuThresholdCount": 0,
            "gpuRecoveryCount": -5
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.fallback.gpuThresholdCount, 1);  // Min 1
    EXPECT_EQ(config.fallback.gpuRecoveryCount, 1);   // Min 1
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackValidation_MonitorIntervalMin10) {
    writeConfig(R"({
        "fallback": {
            "monitorIntervalMs": 5
        }
    })");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.fallback.monitorIntervalMs, 10);  // Min 10ms
}

TEST_F(ConfigLoaderTest, LoadConfigWithFallbackWrongTypes) {
    // Test when fallback is not an object (should be ignored, keep defaults)
    writeConfig(R"({"fallback": "not an object"})");

    AppConfig config;
    bool result = loadAppConfig(testConfigPath, config, false);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.fallback.enabled);
    EXPECT_FLOAT_EQ(config.fallback.gpuThreshold, 80.0f);
}
