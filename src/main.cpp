#include "audio_io.h"
#include "config_loader.h"
#include "convolution_engine.h"
#include "filter_metadata.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

void printUsage(const char* programName) {
    std::cout << "GPU Audio Upsampler - Phase 2" << std::endl;
    std::cout << "Usage: " << programName << " <input.wav> <output.wav> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --filter <path>    Path to filter coefficients .bin file" << std::endl;
    std::cout << "                     (default: data/coefficients/filter_1m_min_phase.bin)"
              << std::endl;
    std::cout << "  --ratio <n>        Upsample ratio (default: 16)" << std::endl;
    std::cout << "  --block <size>     Block size for processing (default: 8192)" << std::endl;
    std::cout << "  --help             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " input_44k.wav output_705k.wav" << std::endl;
    std::cout << "  " << programName << " test.wav upsampled.wav --ratio 16 --block 4096"
              << std::endl;
}

struct Config {
    std::string inputFile;
    std::string outputFile;
    std::string filterPath = std::string(FILTER_PRESET_44K.path);
    int upsampleRatio = FILTER_PRESET_44K.upsampleRatio;
    int blockSize = 8192;
    bool filterOverridden = false;
    bool configLoaded = false;
};

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
        } else if (arg == "--ratio" && i + 1 < argc) {
            config.upsampleRatio = std::stoi(argv[++i]);
        } else if (arg == "--block" && i + 1 < argc) {
            config.blockSize = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU Audio Upsampler - Phase 2" << std::endl;
    std::cout << "  High-Precision Audio Oversampling" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

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
        std::cout << "Step 1: Reading input file..." << std::endl;
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
                          << " Hz input" << std::endl;
                config.upsampleRatio = preset.upsampleRatio;
            }
            std::cout << "Auto-selected filter: " << preset.description << std::endl;
            return true;
        };

        if (!config.filterOverridden) {
            const FilterPreset* targetPreset = nullptr;
            if (inputAudio.sampleRate == FILTER_PRESET_44K.inputSampleRate) {
                targetPreset = &FILTER_PRESET_44K;
            } else if (inputAudio.sampleRate == FILTER_PRESET_48K.inputSampleRate) {
                targetPreset = &FILTER_PRESET_48K;
            } else {
                std::cerr << "Error: Unsupported input sample rate " << inputAudio.sampleRate
                          << " Hz. Supported: 44100 Hz or 48000 Hz." << std::endl;
                return 1;
            }

            if (targetPreset && applyPreset(*targetPreset)) {
                // OK
            } else if (targetPreset == &FILTER_PRESET_48K) {
                std::cerr << "Warning: 48kHz preset filter missing: " << FILTER_PRESET_48K.path
                          << std::endl;
                std::cerr << "To generate: "
                          << "python scripts/generate_filter.py --input-rate 48000 "
                          << "--stopband-start 24000 --passband-end 21500 "
                          << "--output-prefix filter_48k_1m_min_phase" << std::endl;
                if (!applyPreset(FILTER_PRESET_44K)) {
                    std::cerr << "Error: 44.1kHz fallback filter also missing: "
                              << FILTER_PRESET_44K.path << std::endl;
                    return 1;
                }
                std::cout << "Falling back to 44.1kHz preset filter." << std::endl;
            } else {
                std::cerr << "Error: Preset filter file not found: "
                          << (targetPreset ? targetPreset->path : "") << std::endl;
                std::cerr << "Generate it via scripts/generate_filter.py or specify with --filter."
                          << std::endl;
                return 1;
            }
        } else {
            std::cout << "User-specified filter will be used: " << config.filterPath << std::endl;
            if (inputAudio.sampleRate != FILTER_PRESET_44K.inputSampleRate &&
                inputAudio.sampleRate != FILTER_PRESET_48K.inputSampleRate) {
                std::cerr << "Warning: Input sample rate " << inputAudio.sampleRate
                          << " Hz is not validated against provided filter." << std::endl;
            }
        }

        if (!std::filesystem::exists(config.filterPath)) {
            std::cerr << "Error: Filter file not found: " << config.filterPath << std::endl;
            if (inputAudio.sampleRate == FILTER_PRESET_48K.inputSampleRate) {
                std::cerr << "Generate it via: "
                          << "python scripts/generate_filter.py --input-rate 48000 "
                          << "--stopband-start 24000 --passband-end 21500 "
                          << "--output-prefix filter_48k_1m_min_phase" << std::endl;
            } else {
                std::cerr << "Generate it via scripts/generate_filter.py or specify with --filter."
                          << std::endl;
            }
            return 1;
        }

        // Step 2: Initialize GPU upsampler
        std::cout << std::endl << "Step 2: Initializing GPU engine..." << std::endl;
        ConvolutionEngine::GPUUpsampler upsampler;
        if (!upsampler.initialize(config.filterPath, config.upsampleRatio, config.blockSize)) {
            return 1;
        }

        // Step 3: Process audio
        std::cout << std::endl << "Step 3: Processing audio..." << std::endl;

        std::vector<float> outputLeft, outputRight;
        bool success = false;

        if (inputAudio.channels == 1) {
            // Mono input
            std::cout << "  Processing mono channel..." << std::endl;
            success =
                upsampler.processChannel(inputAudio.data.data(), inputAudio.frames, outputLeft);

            // Duplicate to stereo
            outputRight = outputLeft;

        } else if (inputAudio.channels == 2) {
            // Stereo input - separate channels
            std::cout << "  Processing stereo (L/R) channels..." << std::endl;

            std::vector<float> inputLeft(inputAudio.frames);
            std::vector<float> inputRight(inputAudio.frames);

            AudioIO::Utils::interleavedToSeparate(inputAudio.data.data(), inputLeft.data(),
                                                  inputRight.data(), inputAudio.frames);

            success = upsampler.processStereo(inputLeft.data(), inputRight.data(),
                                              inputAudio.frames, outputLeft, outputRight);

        } else {
            std::cerr << "Error: Unsupported channel count: " << inputAudio.channels << std::endl;
            return 1;
        }

        if (!success) {
            std::cerr << "Error: Audio processing failed" << std::endl;
            return 1;
        }

        // Step 4: Interleave stereo output
        std::cout << std::endl << "Step 4: Preparing output..." << std::endl;

        size_t outputFrames = outputLeft.size();
        std::vector<float> outputInterleaved(outputFrames * 2);

        AudioIO::Utils::separateToInterleaved(outputLeft.data(), outputRight.data(),
                                              outputInterleaved.data(), outputFrames);

        // Step 5: Write output WAV file
        std::cout << std::endl << "Step 5: Writing output file..." << std::endl;

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
        std::cout << std::endl << "========================================" << std::endl;
        std::cout << "Processing completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

        const auto& stats = upsampler.getStats();
        double inputDuration = static_cast<double>(inputAudio.frames) / inputAudio.sampleRate;
        double processingSpeed = inputDuration / stats.totalProcessingTime;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Input:  " << inputAudio.frames << " frames @ " << inputAudio.sampleRate
                  << " Hz (" << inputDuration << " sec)" << std::endl;
        std::cout << "Output: " << outputFrames << " frames @ " << outputSampleRate << " Hz"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Performance:" << std::endl;
        std::cout << "  Processing time: " << stats.totalProcessingTime << " sec" << std::endl;
        std::cout << "  Total time:      " << totalElapsed.count() << " sec" << std::endl;
        std::cout << "  Speed:           " << processingSpeed << "x realtime" << std::endl;
        std::cout << std::endl;
        std::cout << "Output file: " << config.outputFile << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
