#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <filesystem>
#include <string>
#include <vector>

constexpr const char* DEFAULT_CONFIG_FILE = "config.json";

// Phase type for FIR filter
enum class PhaseType {
    Minimum,  // Minimum phase: no pre-ringing, frequency-dependent delay (recommended)
    Linear    // Linear phase: pre-ringing present, constant delay, symmetric
};

// Output pipeline mode (Issue #515)
enum class OutputMode { Usb };

struct OutputUsbOptions {
    std::string preferredDevice = "hw:USB";
};

struct OutputConfig {
    OutputMode mode = OutputMode::Usb;
    OutputUsbOptions usb;
};

struct AppConfig {
    std::string alsaDevice = "hw:USB";
    int bufferSize = 262144;
    int periodSize = 32768;
    int upsampleRatio = 16;
    int blockSize = 4096;
    float gain = 1.0f;
    float headroomTarget = 0.92f;  // Peak target (linear 0-1) for headroom + limiter guard
    std::string filterPath = "data/coefficients/filter_44k_16x_2m_linear_phase.bin";
    PhaseType phaseType = PhaseType::Minimum;  // Filter phase type (default: Minimum)

    // Per-family/phase filter paths (quad-phase engine always uses all four)
    std::string filterPath44kMin = "data/coefficients/filter_44k_16x_2m_linear_phase.bin";
    std::string filterPath48kMin = "data/coefficients/filter_48k_16x_2m_linear_phase.bin";
    std::string filterPath44kLinear = "data/coefficients/filter_44k_16x_2m_linear_phase.bin";
    std::string filterPath48kLinear = "data/coefficients/filter_48k_16x_2m_linear_phase.bin";

    // Multi-rate mode: 10 filter configurations (2 rate families × 5 ratios: 1x/2x/4x/8x/16x)
    // Issue #219: Dynamic rate switching with all filters preloaded
    bool multiRateEnabled = false;  // Enable multi-rate mode with all 10 filter FFTs preloaded
    std::string coefficientDir = "data/coefficients";  // Directory containing filter files

    struct LoopbackInputConfig {
        bool enabled = false;
        std::string device = "hw:Loopback,1,0";
        uint32_t sampleRate = 44100;  // 0 = auto/detected (if available)
        uint8_t channels = 2;
        std::string format = "S16_LE";  // Supported: S16_LE, S24_3LE, S32_LE
        uint32_t periodFrames = 1024;
    } loopback;

    // EQ settings
    bool eqEnabled = false;
    std::string eqProfilePath = "";  // Path to EQ profile file (empty = disabled)

    // Partitioned convolution (Issue #351)
    struct PartitionedConvolutionConfig {
        bool enabled = false;
        int fastPartitionTaps = 32768;
        int minPartitionTaps = 32768;
        int maxPartitions = 4;
        int tailFftMultiple = 2;
    } partitionedConvolution;

    // Crossfeed settings (nested struct for clarity)
    struct CrossfeedConfig {
        bool enabled = false;
        std::string headSize = "m";  // "s", "m", "l", "xl"
        std::string hrtfPath = "data/crossfeed/hrtf/";
    } crossfeed;

    // Fallback settings (Issue #139)
    struct FallbackConfig {
        bool enabled = true;                 // Enable dynamic fallback
        float gpuThreshold = 80.0f;          // GPU utilization threshold (%)
        int gpuThresholdCount = 3;           // Consecutive threshold exceedances to trigger
        float gpuRecoveryThreshold = 70.0f;  // Recovery threshold (threshold - 10%)
        int gpuRecoveryCount = 5;            // Consecutive recovery measurements to return
        bool xrunTriggersFallback = true;    // Whether XRUN triggers immediate fallback
        int monitorIntervalMs = 100;         // GPU monitoring interval (milliseconds)
    } fallback;

    OutputConfig output;
};

// Convert string to PhaseType (returns Minimum for invalid input)
PhaseType parsePhaseType(const std::string& str);

// Convert PhaseType to string
const char* phaseTypeToString(PhaseType type);

// Convert string to OutputMode (invalid -> Usb)
OutputMode parseOutputMode(const std::string& str);

// Convert OutputMode to string value ("usb", ...)
const char* outputModeToString(OutputMode mode);

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig,
                   bool verbose = true);

#endif  // CONFIG_LOADER_H
