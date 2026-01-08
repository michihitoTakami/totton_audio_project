#include "daemon/audio/upsampler_builder.h"

#include "audio/eq_parser.h"
#include "audio/eq_to_fir.h"
#include "convolution_engine.h"
#if defined(HAVE_VULKAN_BACKEND)
#include "vulkan/vulkan_streaming_upsampler.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

namespace daemon_audio {
namespace {

bool env_flag(const char* name, bool defaultValue) {
    const char* val = std::getenv(name);
    if (!val) {
        return defaultValue;
    }
    std::string s(val);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
    s.erase(
        std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(),
        s.end());

    if (s == "1" || s == "true" || s == "yes" || s == "on") {
        return true;
    }
    if (s == "0" || s == "false" || s == "no" || s == "off") {
        return false;
    }
    return defaultValue;
}

void resolve_filter_path(AppConfig& config, int inputSampleRate) {
    if (!std::filesystem::exists(config.filterPath)) {
        std::string basePath = config.filterPath;
        size_t dotPos = basePath.rfind('.');
        if (dotPos != std::string::npos) {
            std::string rateSpecificPath = basePath.substr(0, dotPos) + "_" +
                                           std::to_string(inputSampleRate) +
                                           basePath.substr(dotPos);
            if (std::filesystem::exists(rateSpecificPath)) {
                std::cout << "Config: Using sample-rate-specific filter: " << rateSpecificPath
                          << '\n';
                config.filterPath = rateSpecificPath;
            }
        }
    }

    if (inputSampleRate == 48000 && config.filterPath.find("44100") == std::string::npos &&
        config.filterPath.find("48000") == std::string::npos) {
        std::cout << "Warning: Using generic filter with 48kHz input. "
                  << "For optimal quality, generate a 48kHz-optimized filter." << '\n';
    }
}

}  // namespace

UpsamplerBuildResult buildUpsampler(AppConfig& config, int inputSampleRate,
                                    const PartitionRuntime::RuntimeRequest& partitionRequest,
                                    const std::atomic<bool>& runningFlag) {
    UpsamplerBuildResult result{};

    resolve_filter_path(config, inputSampleRate);

    if (config.gpuBackend == GpuBackend::Vulkan) {
#if defined(HAVE_VULKAN_BACKEND)
        std::cout << "Initializing Vulkan upsampler..." << '\n';
        vulkan_backend::VulkanStreamingUpsampler::InitParams params{};
        // Select per-family filter paths (minimum + linear) so phase switching works under Vulkan.
        ConvolutionEngine::RateFamily family = ConvolutionEngine::detectRateFamily(inputSampleRate);
        if (family == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
            family = ConvolutionEngine::RateFamily::RATE_44K;
        }

        std::string minPath = (family == ConvolutionEngine::RateFamily::RATE_48K)
                                  ? config.filterPath48kMin
                                  : config.filterPath44kMin;
        std::string linearPath = (family == ConvolutionEngine::RateFamily::RATE_48K)
                                     ? config.filterPath48kLinear
                                     : config.filterPath44kLinear;

        // Backward compatibility: if quad-phase paths are not configured, fall back to filterPath
        // and derive the counterpart by filename convention.
        if (minPath.empty()) {
            minPath = config.filterPath;
        }
        if (linearPath.empty()) {
            linearPath = config.filterPath;
        }

        if (!std::filesystem::exists(minPath)) {
            std::cerr << "Config error: Vulkan minimum-phase filter not found: " << minPath << '\n';
            return result;
        }
        if (!std::filesystem::exists(linearPath)) {
            std::cerr << "Config warning: Vulkan linear-phase filter not found: " << linearPath
                      << " (phase switching to linear may fail)\n";
        }

        // Provide both families so Vulkan can pre-cache FFTs and allow runtime switching.
        params.filterPathMinimum44k = config.filterPath44kMin;
        params.filterPathMinimum48k = config.filterPath48kMin;
        params.filterPathLinear44k = config.filterPath44kLinear;
        params.filterPathLinear48k = config.filterPath48kLinear;
        params.filterPathMinimum = minPath;
        params.filterPathLinear = linearPath;
        params.initialPhase = config.phaseType;
        params.upsampleRatio = static_cast<uint32_t>(config.upsampleRatio);
        params.blockSize = static_cast<uint32_t>(config.blockSize);
        params.inputRate = static_cast<uint32_t>(inputSampleRate);

        auto upsampler = std::make_unique<vulkan_backend::VulkanStreamingUpsampler>();
        if (!upsampler->initialize(params)) {
            std::cerr << "Failed to initialize Vulkan upsampler" << '\n';
            return result;
        }

        if (!runningFlag.load(std::memory_order_acquire)) {
            std::cout << "Startup interrupted by signal" << '\n';
            result.status = UpsamplerBuildStatus::Interrupted;
            return result;
        }

        result.status = UpsamplerBuildStatus::Success;
        result.currentInputRate = inputSampleRate;
        result.currentOutputRate = static_cast<int>(params.upsampleRatio) * inputSampleRate;
        result.initialRateFamily = ConvolutionEngine::detectRateFamily(inputSampleRate);
        result.upsampler = std::move(upsampler);
        return result;
#else
        std::cerr << "Vulkan backend requested but binary was built without ENABLE_VULKAN=ON"
                  << '\n';
        result.status = UpsamplerBuildStatus::Failure;
        return result;
#endif
    }

    std::cout << "Initializing GPU upsampler..." << '\n';
    auto upsampler = std::make_unique<ConvolutionEngine::GPUUpsampler>();
    upsampler->setPartitionedConvolutionConfig(config.partitionedConvolution);

    bool initSuccess = false;
    ConvolutionEngine::RateFamily initialFamily = ConvolutionEngine::RateFamily::RATE_44K;

    if (config.multiRateEnabled) {
        std::cout << "Multi-rate mode enabled" << '\n';
        std::cout << "  Coefficient directory: " << config.coefficientDir << '\n';

        if (!std::filesystem::exists(config.coefficientDir)) {
            std::cerr << "Config error: Coefficient directory not found: " << config.coefficientDir
                      << '\n';
            return result;
        }

        initSuccess = upsampler->initializeMultiRate(config.coefficientDir, config.blockSize,
                                                     inputSampleRate);

        if (initSuccess) {
            result.currentInputRate = upsampler->getInputSampleRate();
            result.currentOutputRate = upsampler->getOutputSampleRate();
        }
    } else {
        std::cout << "Quad-phase mode enabled" << '\n';

        bool allFilesExist = true;
        for (const auto& path : {config.filterPath44kMin, config.filterPath48kMin,
                                 config.filterPath44kLinear, config.filterPath48kLinear}) {
            if (!std::filesystem::exists(path)) {
                std::cerr << "Config error: Filter file not found: " << path << '\n';
                allFilesExist = false;
            }
        }
        if (!allFilesExist) {
            return result;
        }

        initialFamily = ConvolutionEngine::detectRateFamily(inputSampleRate);
        if (initialFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
            initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
        }

        initSuccess = upsampler->initializeQuadPhase(
            config.filterPath44kMin, config.filterPath48kMin, config.filterPath44kLinear,
            config.filterPath48kLinear, config.upsampleRatio, config.blockSize, initialFamily,
            config.phaseType);
    }

    if (!initSuccess) {
        std::cerr << "Failed to initialize GPU upsampler" << '\n';
        return result;
    }

    if (!runningFlag.load(std::memory_order_acquire)) {
        std::cout << "Startup interrupted by signal" << '\n';
        result.status = UpsamplerBuildStatus::Interrupted;
        return result;
    }

    if (config.multiRateEnabled) {
        std::cout << "GPU upsampler ready (multi-rate mode, " << config.blockSize
                  << " samples/block)" << '\n';
        std::cout << "  Current input rate: " << upsampler->getCurrentInputRate() << " Hz" << '\n';
        std::cout << "  Upsample ratio: " << upsampler->getUpsampleRatio() << "x" << '\n';
        std::cout << "  Output rate: " << upsampler->getOutputSampleRate() << " Hz" << '\n';
    } else {
        std::cout << "GPU upsampler ready (" << config.upsampleRatio << "x upsampling, "
                  << config.blockSize << " samples/block)" << '\n';
    }

    std::cout << "Input sample rate: " << upsampler->getInputSampleRate() << " Hz -> "
              << upsampler->getOutputSampleRate() << " Hz output" << '\n';
    if (!config.multiRateEnabled) {
        std::cout << "Phase type: " << phaseTypeToString(config.phaseType) << '\n';
    }

    if (config.phaseType == PhaseType::Linear) {
        double latencySec = upsampler->getLatencySeconds();
        std::cout << "  WARNING: Linear phase latency: " << latencySec << " seconds ("
                  << upsampler->getLatencySamples() << " samples)" << '\n';
    }

    if (!upsampler->initializeStreaming()) {
        std::cerr << "Failed to initialize streaming mode" << '\n';
        return result;
    }

    bool enableNonBlocking = env_flag("TOTTON_AUDIO_GPU_STREAMING_NONBLOCKING", false);
    upsampler->setStreamingNonBlocking(enableNonBlocking);
    std::cout << "GPU streaming non-blocking: " << (enableNonBlocking ? "enabled" : "disabled")
              << '\n';

    PartitionRuntime::applyPartitionPolicy(partitionRequest, *upsampler, config, "ALSA");

    if (!runningFlag.load(std::memory_order_acquire)) {
        std::cout << "Startup interrupted by signal" << '\n';
        result.status = UpsamplerBuildStatus::Interrupted;
        return result;
    }

    if (config.eqEnabled && !config.eqProfilePath.empty()) {
        std::cout << "Loading EQ profile: " << config.eqProfilePath << '\n';
        EQ::EqProfile eqProfile;
        if (EQ::parseEqFile(config.eqProfilePath, eqProfile)) {
            std::cout << "  EQ: " << eqProfile.name << " (" << eqProfile.bands.size()
                      << " bands, preamp " << eqProfile.preampDb << " dB)" << '\n';

            size_t filterFftSize = upsampler->getFilterFftSize();
            size_t fullFftSize = upsampler->getFullFftSize();
            double outputSampleRate = static_cast<double>(inputSampleRate) * config.upsampleRatio;
            auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                            outputSampleRate, eqProfile);
            double eqMax = 0.0;
            double eqMin = std::numeric_limits<double>::infinity();
            for (double v : eqMagnitude) {
                eqMax = std::max(eqMax, v);
                eqMin = std::min(eqMin, v);
            }
            std::cout << "  EQ magnitude stats: max=" << eqMax << " ("
                      << 20.0 * std::log10(std::max(eqMax, 1e-30)) << " dB), min=" << eqMin << '\n';

            if (!upsampler->applyEqMagnitude(eqMagnitude)) {
                std::cerr << "  EQ: Failed to apply frequency response" << '\n';
            }
        } else {
            std::cerr << "  EQ: Failed to parse profile: " << config.eqProfilePath << '\n';
        }
    }

    result.status = UpsamplerBuildStatus::Success;
    result.upsampler = std::move(upsampler);
    result.initialRateFamily = initialFamily;
    return result;
}

}  // namespace daemon_audio
