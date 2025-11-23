#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <filesystem>
#include <string>

constexpr const char* DEFAULT_CONFIG_FILE = "config.json";

// Phase type for FIR filter
enum class PhaseType {
    Minimum,  // Minimum phase: no pre-ringing, frequency-dependent delay (recommended)
    Linear    // Linear phase: pre-ringing present, constant delay, symmetric
};

struct AppConfig {
    std::string alsaDevice = "hw:USB";
    int bufferSize = 262144;
    int periodSize = 32768;
    int upsampleRatio = 16;
    int blockSize = 4096;
    float gain = 16.0f;
    std::string filterPath = "data/coefficients/filter_1m_min_phase.bin";
    int inputSampleRate = 44100;               // Input sample rate (44100 or 48000)
    PhaseType phaseType = PhaseType::Minimum;  // Filter phase type (default: Minimum)

    // EQ settings
    bool eqEnabled = false;
    std::string eqProfilePath = "";  // Path to EQ profile file (empty = disabled)
};

// Convert string to PhaseType (returns Minimum for invalid input)
PhaseType parsePhaseType(const std::string& str);

// Convert PhaseType to string
const char* phaseTypeToString(PhaseType type);

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig,
                   bool verbose = true);

#endif  // CONFIG_LOADER_H
