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
    float gain = 1.0f;
    std::string filterPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    PhaseType phaseType = PhaseType::Minimum;  // Filter phase type (default: Minimum)

    // Quad-phase mode: 4 filter paths (2 rate families × 2 phase types)
    bool quadPhaseEnabled = false;  // Enable quad-phase mode with all 4 filter FFTs preloaded
    std::string filterPath44kMin = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
    std::string filterPath48kMin = "data/coefficients/filter_48k_2m_min_phase.bin";
    std::string filterPath44kLinear = "data/coefficients/filter_44k_16x_2m_linear.bin";
    std::string filterPath48kLinear = "data/coefficients/filter_48k_16x_2m_linear.bin";

    // EQ settings
    bool eqEnabled = false;
    std::string eqProfilePath = "";  // Path to EQ profile file (empty = disabled)

    // Crossfeed settings (nested struct for clarity)
    struct CrossfeedConfig {
        bool enabled = false;
        std::string headSize = "m";  // "s", "m", "l", "xl"
        std::string hrtfPath = "data/crossfeed/hrtf/";
    } crossfeed;
};

// Convert string to PhaseType (returns Minimum for invalid input)
PhaseType parsePhaseType(const std::string& str);

// Convert PhaseType to string
const char* phaseTypeToString(PhaseType type);

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig,
                   bool verbose = true);

#endif  // CONFIG_LOADER_H
