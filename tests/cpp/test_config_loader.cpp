/**
 * @file test_config_loader.cpp
 * @brief Unit tests for config loader (JSON configuration)
 */

#include "config_loader.h"

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
    EXPECT_EQ(config.filterPath, "data/coefficients/filter_44k_2m_min_phase.bin");
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
