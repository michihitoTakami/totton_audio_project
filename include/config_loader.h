#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <filesystem>
#include <string>

constexpr const char* DEFAULT_CONFIG_FILE = "config.json";

struct AppConfig {
    std::string alsaDevice = "hw:USB";
    int bufferSize = 262144;
    int periodSize = 32768;
    int upsampleRatio = 16;
    int blockSize = 4096;
    float gain = 16.0f;
    std::string filterPath = "data/coefficients/filter_1m_min_phase.bin";
    int inputSampleRate = 44100;  // Input sample rate (44100 or 48000)

    // EQ settings
    bool eqEnabled = false;
    std::string eqProfilePath = "";  // Path to EQ profile file (empty = disabled)
};

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig, bool verbose = true);

#endif // CONFIG_LOADER_H
