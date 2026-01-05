/**
 * Delimiter WAV-to-WAV processor
 *
 * Processes audio through the C++ De-limiter inference backend
 * (same code path as production daemon) for validation and comparison.
 *
 * Usage:
 *   delimiter_wav_to_wav --input in.wav --output out.wav --model delimiter.onnx
 */

#include "audio/audio_utils.h"
#include "audio/overlap_add.h"
#include "core/config_loader.h"
#include "delimiter/inference_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sndfile.h>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string inputPath;
    std::string outputPath;
    std::string modelPath;
    std::string provider = "cuda";
    float chunkSec = 4.0f;
    float overlapSec = 0.25f;
    int expectedSampleRate = 44100;
    int intraOpThreads = 0;
    bool verbose = false;
};

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --input <path>       Input WAV file (required)\n"
              << "  --output <path>      Output WAV file (required)\n"
              << "  --model <path>       ONNX model path (required)\n"
              << "  --provider <name>    ORT provider: cpu, cuda, tensorrt (default: cuda)\n"
              << "  --chunk-sec <float>  Chunk size in seconds (default: 4.0)\n"
              << "  --overlap-sec <float> Overlap size in seconds (default: 0.25)\n"
              << "  --sample-rate <int>  Expected sample rate (default: 44100)\n"
              << "  --threads <int>      Intra-op threads (default: 0 = auto)\n"
              << "  --verbose            Enable verbose output\n";
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            args.inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.outputPath = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            args.modelPath = argv[++i];
        } else if (arg == "--provider" && i + 1 < argc) {
            args.provider = argv[++i];
        } else if (arg == "--chunk-sec" && i + 1 < argc) {
            args.chunkSec = std::stof(argv[++i]);
        } else if (arg == "--overlap-sec" && i + 1 < argc) {
            args.overlapSec = std::stof(argv[++i]);
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            args.expectedSampleRate = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.intraOpThreads = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (args.inputPath.empty() || args.outputPath.empty() || args.modelPath.empty()) {
        std::cerr << "Error: --input, --output, and --model are required\n";
        return false;
    }
    return true;
}

std::vector<float> makeRaisedCosineFade(std::size_t length) {
    std::vector<float> fade(length);
    for (std::size_t i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(length);
        fade[i] = static_cast<float>(0.5 - 0.5 * std::cos(M_PI * t));
    }
    return fade;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }

    // Open input file
    SF_INFO sfInfo{};
    SNDFILE* inFile = sf_open(args.inputPath.c_str(), SFM_READ, &sfInfo);
    if (!inFile) {
        std::cerr << "Error: Cannot open input file: " << args.inputPath << "\n";
        return 1;
    }

    if (sfInfo.channels != 2) {
        std::cerr << "Error: Only stereo files are supported (got " << sfInfo.channels
                  << " channels)\n";
        sf_close(inFile);
        return 1;
    }

    const int sampleRate = sfInfo.samplerate;
    const sf_count_t totalFrames = sfInfo.frames;

    if (args.verbose) {
        std::cout << "[input] " << args.inputPath << "\n"
                  << "  sample_rate: " << sampleRate << " Hz\n"
                  << "  frames: " << totalFrames << "\n"
                  << "  duration: " << static_cast<double>(totalFrames) / sampleRate << " sec\n";
    }

    // Read entire file (interleaved)
    std::vector<float> interleavedInput(static_cast<std::size_t>(totalFrames) * 2);
    sf_count_t framesRead = sf_readf_float(inFile, interleavedInput.data(), totalFrames);
    sf_close(inFile);

    if (framesRead != totalFrames) {
        std::cerr << "Warning: Read " << framesRead << " frames instead of " << totalFrames << "\n";
    }

    // Deinterleave to planar
    std::vector<float> inputLeft(static_cast<std::size_t>(framesRead));
    std::vector<float> inputRight(static_cast<std::size_t>(framesRead));
    for (std::size_t i = 0; i < static_cast<std::size_t>(framesRead); ++i) {
        inputLeft[i] = interleavedInput[i * 2];
        inputRight[i] = interleavedInput[i * 2 + 1];
    }

    // Create delimiter backend
    AppConfig::DelimiterConfig delimConfig;
    delimConfig.enabled = true;
    delimConfig.backend = "ort";
    delimConfig.chunkSec = args.chunkSec;
    delimConfig.overlapSec = args.overlapSec;
    delimConfig.expectedSampleRate = static_cast<uint32_t>(args.expectedSampleRate);
    delimConfig.ort.modelPath = args.modelPath;
    delimConfig.ort.provider = args.provider;
    delimConfig.ort.intraOpThreads = args.intraOpThreads;

    auto backend = delimiter::createDelimiterInferenceBackend(delimConfig);
    if (!backend) {
        std::cerr << "Error: Failed to create delimiter backend\n";
        return 1;
    }

    if (args.verbose) {
        std::cout << "[backend] " << backend->name() << "\n"
                  << "  expected_sample_rate: " << backend->expectedSampleRate() << " Hz\n";
    }

    // Chunking parameters
    const std::size_t chunkFrames =
        static_cast<std::size_t>(std::round(args.chunkSec * static_cast<float>(sampleRate)));
    const std::size_t overlapFrames =
        static_cast<std::size_t>(std::round(args.overlapSec * static_cast<float>(sampleRate)));
    const std::size_t hopFrames = chunkFrames - overlapFrames;

    if (chunkFrames == 0 || overlapFrames >= chunkFrames) {
        std::cerr << "Error: Invalid chunk/overlap parameters\n";
        return 1;
    }

    if (args.verbose) {
        std::cout << "[chunking]\n"
                  << "  chunk_frames: " << chunkFrames << "\n"
                  << "  overlap_frames: " << overlapFrames << "\n"
                  << "  hop_frames: " << hopFrames << "\n";
    }

    // Prepare overlap-add
    std::vector<float> fadeIn = makeRaisedCosineFade(overlapFrames);
    std::vector<float> outputLeft;
    std::vector<float> outputRight;
    outputLeft.reserve(static_cast<std::size_t>(framesRead) + chunkFrames);
    outputRight.reserve(static_cast<std::size_t>(framesRead) + chunkFrames);

    std::vector<float> prevTailLeft(overlapFrames, 0.0f);
    std::vector<float> prevTailRight(overlapFrames, 0.0f);
    bool hasPrevChunk = false;

    // Process chunks
    auto startTime = std::chrono::steady_clock::now();
    std::size_t chunksProcessed = 0;
    double totalInferenceMs = 0.0;

    for (std::size_t offset = 0; offset < static_cast<std::size_t>(framesRead);) {
        // Prepare chunk
        std::size_t chunkStart = offset;
        std::size_t chunkEnd = std::min(offset + chunkFrames, static_cast<std::size_t>(framesRead));
        std::size_t actualChunkFrames = chunkEnd - chunkStart;

        std::vector<float> chunkLeft(chunkFrames, 0.0f);
        std::vector<float> chunkRight(chunkFrames, 0.0f);
        std::copy(inputLeft.begin() + static_cast<std::ptrdiff_t>(chunkStart),
                  inputLeft.begin() + static_cast<std::ptrdiff_t>(chunkEnd), chunkLeft.begin());
        std::copy(inputRight.begin() + static_cast<std::ptrdiff_t>(chunkStart),
                  inputRight.begin() + static_cast<std::ptrdiff_t>(chunkEnd), chunkRight.begin());

        // Process through backend
        std::vector<float> outLeft, outRight;
        auto inferStart = std::chrono::steady_clock::now();
        auto result = backend->process(
            delimiter::StereoPlanarView{chunkLeft.data(), chunkRight.data(), chunkFrames}, outLeft,
            outRight);
        auto inferEnd = std::chrono::steady_clock::now();
        double inferMs =
            std::chrono::duration_cast<std::chrono::duration<double>>(inferEnd - inferStart)
                .count() *
            1000.0;
        totalInferenceMs += inferMs;

        if (result.status != delimiter::InferenceStatus::Ok) {
            std::cerr << "Warning: Inference failed at chunk " << chunksProcessed << ": "
                      << result.message << "\n";
            // Use input as fallback
            outLeft = chunkLeft;
            outRight = chunkRight;
        }

        // Ensure output size matches input
        if (outLeft.size() != chunkFrames || outRight.size() != chunkFrames) {
            std::cerr << "Warning: Output size mismatch at chunk " << chunksProcessed
                      << " (expected " << chunkFrames << ", got " << outLeft.size() << ")\n";
            outLeft.resize(chunkFrames, 0.0f);
            outRight.resize(chunkFrames, 0.0f);
        }

        // Overlap-add
        if (!hasPrevChunk) {
            // First chunk: output hop region directly
            for (std::size_t i = 0; i < hopFrames && i < actualChunkFrames; ++i) {
                outputLeft.push_back(outLeft[i]);
                outputRight.push_back(outRight[i]);
            }
        } else {
            // Crossfade overlap region with previous tail
            for (std::size_t i = 0; i < overlapFrames; ++i) {
                float fadeOut = fadeIn[overlapFrames - 1 - i];
                float blendedL = prevTailLeft[i] * fadeOut + outLeft[i] * fadeIn[i];
                float blendedR = prevTailRight[i] * fadeOut + outRight[i] * fadeIn[i];
                outputLeft.push_back(blendedL);
                outputRight.push_back(blendedR);
            }
            // Output non-overlapped hop region
            for (std::size_t i = overlapFrames; i < hopFrames && i < actualChunkFrames; ++i) {
                outputLeft.push_back(outLeft[i]);
                outputRight.push_back(outRight[i]);
            }
        }

        // Save tail for next overlap
        for (std::size_t i = 0; i < overlapFrames; ++i) {
            std::size_t srcIdx = hopFrames + i;
            prevTailLeft[i] = (srcIdx < outLeft.size()) ? outLeft[srcIdx] : 0.0f;
            prevTailRight[i] = (srcIdx < outRight.size()) ? outRight[srcIdx] : 0.0f;
        }

        hasPrevChunk = true;
        chunksProcessed++;
        offset += hopFrames;

        if (args.verbose && chunksProcessed % 10 == 0) {
            std::cout << "  Processed " << chunksProcessed << " chunks...\r" << std::flush;
        }
    }

    // Flush remaining tail
    for (std::size_t i = 0; i < overlapFrames; ++i) {
        outputLeft.push_back(prevTailLeft[i]);
        outputRight.push_back(prevTailRight[i]);
    }

    auto endTime = std::chrono::steady_clock::now();
    double totalSec =
        std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime).count();
    double audioDuration = static_cast<double>(framesRead) / sampleRate;
    double rtf = totalSec / audioDuration;

    if (args.verbose) {
        std::cout << "\n[performance]\n"
                  << "  chunks: " << chunksProcessed << "\n"
                  << "  total_time: " << totalSec << " sec\n"
                  << "  inference_time: " << totalInferenceMs / 1000.0 << " sec\n"
                  << "  rtf: " << rtf << " (lower is faster)\n";
    }

    // Trim output to match input length
    std::size_t outputFrames = std::min(outputLeft.size(), static_cast<std::size_t>(framesRead));
    outputLeft.resize(outputFrames);
    outputRight.resize(outputFrames);

    // Interleave for output
    std::vector<float> interleavedOutput(outputFrames * 2);
    for (std::size_t i = 0; i < outputFrames; ++i) {
        interleavedOutput[i * 2] = outputLeft[i];
        interleavedOutput[i * 2 + 1] = outputRight[i];
    }

    // Write output file
    SF_INFO outInfo{};
    outInfo.samplerate = sampleRate;
    outInfo.channels = 2;
    outInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* outFile = sf_open(args.outputPath.c_str(), SFM_WRITE, &outInfo);
    if (!outFile) {
        std::cerr << "Error: Cannot create output file: " << args.outputPath << "\n";
        return 1;
    }

    sf_count_t framesWritten =
        sf_writef_float(outFile, interleavedOutput.data(), static_cast<sf_count_t>(outputFrames));
    sf_close(outFile);

    if (args.verbose) {
        std::cout << "[output] " << args.outputPath << "\n"
                  << "  frames_written: " << framesWritten << "\n";
    }

    std::cout << "Processed " << args.inputPath << " -> " << args.outputPath << "\n"
              << "  Duration: " << audioDuration << " sec\n"
              << "  RTF: " << rtf << " (" << (1.0 / rtf) << "x realtime)\n";

    return 0;
}
