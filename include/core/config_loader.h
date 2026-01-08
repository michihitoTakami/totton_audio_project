#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <cstdint>
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

// Headroom gain computation mode
enum class HeadroomMode { PerFilter, FamilyMax };

struct OutputUsbOptions {
    std::string preferredDevice = "hw:USB";
};

struct OutputConfig {
    OutputMode mode = OutputMode::Usb;
    OutputUsbOptions usb;
};

enum class GpuBackend { Cuda, Vulkan };

struct AppConfig {
    std::string alsaDevice = "hw:USB";
    int bufferSize = 262144;
    int periodSize = 32768;
    int upsampleRatio = 16;
    int blockSize = 4096;
    float gain = 1.0f;
    float headroomTarget = 0.92f;  // Peak target (linear 0-1) for headroom + limiter guard
    HeadroomMode headroomMode = HeadroomMode::FamilyMax;  // Gain calc: per-filter or family-max
    std::string filterPath = "data/coefficients/filter_44k_16x_640k_linear_phase.bin";
    PhaseType phaseType = PhaseType::Minimum;  // Filter phase type (default: Minimum)
    GpuBackend gpuBackend = GpuBackend::Cuda;  // Backend selection: CUDA (default) or Vulkan

    // Per-family/phase filter paths (quad-phase engine always uses all four)
    std::string filterPath44kMin = "data/coefficients/filter_44k_16x_640k_linear_phase.bin";
    std::string filterPath48kMin = "data/coefficients/filter_48k_16x_640k_linear_phase.bin";
    std::string filterPath44kLinear = "data/coefficients/filter_44k_16x_640k_linear_phase.bin";
    std::string filterPath48kLinear = "data/coefficients/filter_48k_16x_640k_linear_phase.bin";

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

    // I2S input (Jetson RX) configuration (Issue #906 / Epic #838)
    // Note: sampleRate can be 0 to mean "use current/negotiated rate" (future: #824).
    // For MVP, 44.1k/48k family rates are expected.
    struct I2sInputConfig {
        bool enabled = false;
        std::string device = "hw:I2S";
        uint32_t sampleRate = 0;        // 0 = follow runtime/negotiated rate (default)
        uint8_t channels = 2;           // Typically 2ch
        std::string format = "S32_LE";  // Typical for 24-in-32 slot I2S (signed)
        uint32_t periodFrames = 1024;
    } i2s;

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
        std::string headSize = "m";  // "xs", "s", "m", "l", "xl"
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

    // De-limiter inference backend selection (Issue #1017 / Epic #1006)
    // Notes:
    // - This config only selects/parameterizes the backend. The high-latency worker path that
    //   calls it is implemented in follow-up issues (#1010/#1014).
    struct DelimiterConfig {
        bool enabled = false;

        // Backend name (case-insensitive):
        // - "bypass": no-op (identity), used for wiring/health-check
        // - "ort": ONNX Runtime backend (expected to run out-of-process or optional dependency)
        std::string backend = "bypass";

        // High-latency worker chunking parameters (Fix #1009 / #1010)
        float chunkSec = 4.0f;
        float overlapSec = 0.25f;

        struct OrtConfig {
            // Path to ONNX model file.
            std::string modelPath = "";

            // Execution provider (case-insensitive): "cpu" | "cuda" | "tensorrt"
            std::string provider = "cpu";

            // Optional tuning (left as future extension; defaults are safe)
            int intraOpThreads = 0;  // 0 = ORT default
        } ort;

        // Some backends require fixed input sample rate (De-limiter supports 44.1k/48k).
        uint32_t expectedSampleRate = 44100;
    } delimiter;

    OutputConfig output;
};

// Convert string to PhaseType (returns Minimum for invalid input)
PhaseType parsePhaseType(const std::string& str);

// Convert PhaseType to string
const char* phaseTypeToString(PhaseType type);

// Convert string to HeadroomMode (returns FamilyMax for invalid input)
HeadroomMode parseHeadroomMode(const std::string& str);

// Convert HeadroomMode to string ("per_filter", "family_max")
const char* headroomModeToString(HeadroomMode mode);

// Convert string to OutputMode (invalid -> Usb)
OutputMode parseOutputMode(const std::string& str);

// Convert OutputMode to string value ("usb", ...)
const char* outputModeToString(OutputMode mode);

// Convert backend string to enum (defaults to Cuda on invalid)
GpuBackend parseGpuBackend(const std::string& str);
const char* gpuBackendToString(GpuBackend backend);

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig,
                   bool verbose = true);

#endif  // CONFIG_LOADER_H
