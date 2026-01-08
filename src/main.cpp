#include "audio/audio_io.h"
#include "audio/eq_parser.h"
#include "audio/eq_to_fir.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/filter_metadata.h"
#include "logging/logger.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

void printUsage(const char* programName) {
    std::cout << "GPU Audio Upsampler - Phase 2" << '\n';
    std::cout << "Usage: " << programName << " <input.wav> <output.wav> [options]" << '\n';
    std::cout << '\n';
    std::cout << "Options:" << '\n';
    std::cout << "  --filter <path>    Path to filter coefficients .bin file" << '\n';
    std::cout
        << "                     (default: data/coefficients/filter_44k_16x_640k_linear_phase.bin)"
        << '\n';
    std::cout << "  --eq <path>        Path to EQ profile file (optional)" << '\n';
    std::cout << "  --ratio <n>        Upsample ratio (default: 16)" << '\n';
    std::cout << "  --block <size>     Block size for processing (default: 8192)" << '\n';
    std::cout << "  --help             Show this help message" << '\n';
    std::cout << '\n';
    std::cout << "Examples:" << '\n';
    std::cout << "  " << programName << " input_44k.wav output_705k.wav" << '\n';
    std::cout << "  " << programName << " test.wav upsampled.wav --ratio 16 --block 4096" << '\n';
}

struct Config {
    std::string inputFile;
    std::string outputFile;
    std::string filterPath = std::string(FILTER_PRESET_44K.path);
    std::string eqPath;
    int upsampleRatio = FILTER_PRESET_44K.upsampleRatio;
    int blockSize = 8192;
    bool filterOverridden = false;
    bool configLoaded = false;
};

static float computePeakLinear(const float* samples, size_t count) {
    float peak = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float v = std::fabs(samples[i]);
        if (v > peak) {
            peak = v;
        }
    }
    return peak;
}

static double linearToDbfs(double value) {
    if (value <= 0.0) {
        return -200.0;
    }
    return 20.0 * std::log10(value);
}

static void printStereoPeaks(const char* label, const std::vector<float>& interleaved) {
    if (interleaved.empty()) {
        std::cout << label << ": (empty)" << '\n';
        return;
    }
    float peak = computePeakLinear(interleaved.data(), interleaved.size());
    std::cout << label << ": peak=" << std::fixed << std::setprecision(6) << peak << " ("
              << std::setprecision(2) << linearToDbfs(peak) << " dBFS)" << '\n';
}

bool parseArguments(int argc, char* argv[], Config& config) {
    if (argc < 3) {
        return false;
    }

    config.inputFile = argv[1];
    config.outputFile = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--filter" && i + 1 < argc) {
            config.filterPath = argv[++i];
            config.filterOverridden = true;
        } else if (arg == "--eq" && i + 1 < argc) {
            config.eqPath = argv[++i];
        } else if (arg == "--ratio" && i + 1 < argc) {
            config.upsampleRatio = std::stoi(argv[++i]);
        } else if (arg == "--block" && i + 1 < argc) {
            config.blockSize = std::stoi(argv[++i]);
        } else {
            LOG_ERROR("Unknown option: {}", arg);
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    gpu_upsampler::logging::initializeEarly();

    std::cout << "========================================" << '\n';
    std::cout << "  GPU Audio Upsampler - Phase 2" << '\n';
    std::cout << "  High-Precision Audio Oversampling" << '\n';
    std::cout << "========================================" << '\n';
    std::cout << '\n';

    AppConfig appConfig;
    Config config;
    if (loadAppConfig(DEFAULT_CONFIG_FILE, appConfig)) {
        config.filterPath = appConfig.filterPath;
        config.upsampleRatio = appConfig.upsampleRatio;
        config.blockSize = appConfig.blockSize;
        if (!appConfig.filterPath.empty() &&
            appConfig.filterPath != std::string(FILTER_PRESET_44K.path)) {
            config.filterOverridden = true;
        }
        config.configLoaded = true;
    }

    // Parse command line arguments
    if (!parseArguments(argc, argv, config)) {
        printUsage(argv[0]);
        return 1;
    }
    if (config.blockSize <= 0) {
        config.blockSize = 8192;
    }
    if (config.upsampleRatio <= 0) {
        config.upsampleRatio = FILTER_PRESET_44K.upsampleRatio;
    }

    try {
        auto totalStartTime = std::chrono::high_resolution_clock::now();

        // Step 1: Read input WAV file
        std::cout << "Step 1: Reading input file..." << '\n';
        AudioIO::WavReader reader;
        if (!reader.open(config.inputFile)) {
            return 1;
        }

        AudioIO::AudioFile inputAudio;
        if (!reader.readAll(inputAudio)) {
            return 1;
        }
        reader.close();

        // Validate and select filter based on sample rate (unless user overrides)
        auto applyPreset = [&](const FilterPreset& preset) -> bool {
            if (!std::filesystem::exists(preset.path)) {
                return false;
            }
            config.filterPath = std::string(preset.path);
            if (config.upsampleRatio != preset.upsampleRatio) {
                std::cout << "Info: Overriding upsample ratio to preset value "
                          << preset.upsampleRatio << "x for " << inputAudio.sampleRate
                          << " Hz input" << '\n';
                config.upsampleRatio = preset.upsampleRatio;
            }
            std::cout << "Auto-selected filter: " << preset.description << '\n';
            return true;
        };

        if (!config.filterOverridden) {
            const FilterPreset* targetPreset = nullptr;
            if (inputAudio.sampleRate == FILTER_PRESET_44K.inputSampleRate) {
                targetPreset = &FILTER_PRESET_44K;
            } else if (inputAudio.sampleRate == FILTER_PRESET_48K.inputSampleRate) {
                targetPreset = &FILTER_PRESET_48K;
            } else {
                LOG_ERROR("Unsupported input sample rate {} Hz. Supported: 44100 Hz or 48000 Hz.",
                          inputAudio.sampleRate);
                return 1;
            }

            if (targetPreset && applyPreset(*targetPreset)) {
                // OK
            } else if (targetPreset == &FILTER_PRESET_48K) {
                LOG_WARN("48kHz preset filter missing: {}", FILTER_PRESET_48K.path);
                LOG_INFO(
                    "To generate: python scripts/filters/generate_minimum_phase.py --input-rate "
                    "48000 "
                    "--stopband-start 24000 --passband-end 21500 "
                    "--output-prefix filter_48k_16x_640k_min_phase");
                if (!applyPreset(FILTER_PRESET_44K)) {
                    LOG_ERROR("44.1kHz fallback filter also missing: {}", FILTER_PRESET_44K.path);
                    return 1;
                }
                std::cout << "Falling back to 44.1kHz preset filter." << '\n';
            } else {
                LOG_ERROR("Preset filter file not found: {}",
                          (targetPreset ? targetPreset->path : ""));
                LOG_INFO(
                    "Generate it via scripts/filters/generate_minimum_phase.py or specify with "
                    "--filter.");
                return 1;
            }
        } else {
            std::cout << "User-specified filter will be used: " << config.filterPath << '\n';
            if (inputAudio.sampleRate != FILTER_PRESET_44K.inputSampleRate &&
                inputAudio.sampleRate != FILTER_PRESET_48K.inputSampleRate) {
                LOG_WARN("Input sample rate {} Hz is not validated against provided filter.",
                         inputAudio.sampleRate);
            }
        }

        if (!std::filesystem::exists(config.filterPath)) {
            LOG_ERROR("Filter file not found: {}", config.filterPath);
            if (inputAudio.sampleRate == FILTER_PRESET_48K.inputSampleRate) {
                LOG_INFO(
                    "Generate it via: python scripts/filters/generate_minimum_phase.py "
                    "--input-rate 48000 "
                    "--stopband-start 24000 --passband-end 21500 "
                    "--output-prefix filter_48k_16x_640k_min_phase");
            } else {
                LOG_INFO(
                    "Generate it via scripts/filters/generate_minimum_phase.py or specify with "
                    "--filter.");
            }
            return 1;
        }

        // Step 2: Initialize GPU upsampler
        std::cout << '\n' << "Step 2: Initializing GPU engine..." << '\n';
        ConvolutionEngine::GPUUpsampler upsampler;
        if (!upsampler.initialize(config.filterPath, config.upsampleRatio, config.blockSize)) {
            return 1;
        }

        // Optional: Apply EQ profile
        if (!config.eqPath.empty()) {
            std::cout << '\n' << "Step 2b: Loading EQ profile..." << '\n';
            EQ::EqProfile eqProfile;
            if (!EQ::parseEqFile(config.eqPath, eqProfile)) {
                LOG_ERROR("Failed to parse EQ file: {}", config.eqPath);
                return 1;
            }
            std::cout << "  EQ: " << eqProfile.name << " (" << eqProfile.bands.size()
                      << " bands, preamp " << eqProfile.preampDb << " dB)" << '\n';

            size_t filterFftSize = upsampler.getFilterFftSize();
            size_t fullFftSize = upsampler.getFullFftSize();
            double outputSampleRate = static_cast<double>(inputAudio.sampleRate) *
                                      static_cast<double>(config.upsampleRatio);
            auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                            outputSampleRate, eqProfile);

            double maxMag = 0.0;
            double minMag = std::numeric_limits<double>::infinity();
            for (double v : eqMagnitude) {
                maxMag = std::max(maxMag, v);
                minMag = std::min(minMag, v);
            }
            std::cout << "  EQ magnitude: max=" << std::setprecision(6) << maxMag << " ("
                      << std::setprecision(2) << 20.0 * std::log10(std::max(maxMag, 1e-30))
                      << " dB), min=" << std::setprecision(6) << minMag << '\n';

            if (!upsampler.applyEqMagnitude(eqMagnitude)) {
                LOG_ERROR("Failed to apply EQ magnitude");
                return 1;
            }
        }

        // Step 3: Process audio
        std::cout << '\n' << "Step 3: Processing audio..." << '\n';

        std::vector<float> outputLeft, outputRight;
        bool success = false;

        if (inputAudio.channels == 1) {
            // Mono input
            std::cout << "  Processing mono channel..." << '\n';
            success =
                upsampler.processChannel(inputAudio.data.data(), inputAudio.frames, outputLeft);

            // Duplicate to stereo
            outputRight = outputLeft;

        } else if (inputAudio.channels == 2) {
            // Stereo input - separate channels
            std::cout << "  Processing stereo (L/R) channels..." << '\n';

            std::vector<float> inputLeft(inputAudio.frames);
            std::vector<float> inputRight(inputAudio.frames);

            AudioIO::Utils::interleavedToSeparate(inputAudio.data.data(), inputLeft.data(),
                                                  inputRight.data(), inputAudio.frames);

            success = upsampler.processStereo(inputLeft.data(), inputRight.data(),
                                              inputAudio.frames, outputLeft, outputRight);

        } else {
            LOG_ERROR("Unsupported channel count: {}", inputAudio.channels);
            return 1;
        }

        if (!success) {
            LOG_ERROR("Audio processing failed");
            return 1;
        }

        // Step 4: Interleave stereo output
        std::cout << '\n' << "Step 4: Preparing output..." << '\n';

        size_t outputFrames = outputLeft.size();
        std::vector<float> outputInterleaved(outputFrames * 2);

        AudioIO::Utils::separateToInterleaved(outputLeft.data(), outputRight.data(),
                                              outputInterleaved.data(), outputFrames);

        std::cout << '\n' << "Level check (peak dBFS):" << '\n';
        printStereoPeaks("  Input ", inputAudio.data);
        printStereoPeaks("  Output", outputInterleaved);

        // Step 5: Write output WAV file
        std::cout << '\n' << "Step 5: Writing output file..." << '\n';

        int outputSampleRate = inputAudio.sampleRate * config.upsampleRatio;

        AudioIO::AudioFile outputAudio;
        outputAudio.data = std::move(outputInterleaved);
        outputAudio.sampleRate = outputSampleRate;
        outputAudio.channels = 2;
        outputAudio.frames = outputFrames;

        AudioIO::WavWriter writer;
        if (!writer.open(config.outputFile, outputSampleRate, 2)) {
            return 1;
        }

        if (!writer.writeAll(outputAudio)) {
            return 1;
        }
        writer.close();

        auto totalEndTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalElapsed = totalEndTime - totalStartTime;

        // Print statistics
        std::cout << '\n' << "========================================" << '\n';
        std::cout << "Processing completed successfully!" << '\n';
        std::cout << "========================================" << '\n';

        const auto& stats = upsampler.getStats();
        double inputDuration = static_cast<double>(inputAudio.frames) / inputAudio.sampleRate;
        double processingSpeed = inputDuration / stats.totalProcessingTime;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Input:  " << inputAudio.frames << " frames @ " << inputAudio.sampleRate
                  << " Hz (" << inputDuration << " sec)" << '\n';
        std::cout << "Output: " << outputFrames << " frames @ " << outputSampleRate << " Hz"
                  << '\n';
        std::cout << '\n';
        std::cout << "Performance:" << '\n';
        std::cout << "  Processing time: " << stats.totalProcessingTime << " sec" << '\n';
        std::cout << "  Total time:      " << totalElapsed.count() << " sec" << '\n';
        std::cout << "  Speed:           " << processingSpeed << "x realtime" << '\n';
        std::cout << '\n';
        std::cout << "Output file: " << config.outputFile << '\n';
        std::cout << "========================================" << '\n';

        return 0;

    } catch (const std::exception& e) {
        LOG_CRITICAL("Fatal error: {}", e.what());
        return 1;
    }
}
