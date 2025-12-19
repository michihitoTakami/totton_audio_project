#include "daemon/audio/crossfeed_manager.h"

#include "core/daemon_constants.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace daemon_audio {
namespace {

size_t computeStreamBufferCapacity(const AppConfig& config, size_t streamValidInputPerBlock) {
    using namespace DaemonConstants;
    size_t frames = static_cast<size_t>(DEFAULT_BLOCK_SIZE);
    if (config.blockSize > 0) {
        frames = std::max(frames, static_cast<size_t>(config.blockSize));
    }
    if (config.periodSize > 0) {
        frames = std::max(frames, static_cast<size_t>(config.periodSize));
    }
    if (config.loopback.periodFrames > 0) {
        frames = std::max(frames, static_cast<size_t>(config.loopback.periodFrames));
    }
    frames = std::max(frames, streamValidInputPerBlock);
    // 2x safety margin for bursty upstream (no reallocation in RT path)
    return frames * 2;
}

void allocateCrossfeedBuffers(daemon_app::CrossfeedState& state, const AppConfig& config,
                              size_t streamValidInputPerBlock, size_t validOutputPerBlock,
                              bool logCapacity) {
    size_t bufferCapacity = computeStreamBufferCapacity(config, streamValidInputPerBlock);
    state.cfStreamInputLeft.resize(bufferCapacity, 0.0f);
    state.cfStreamInputRight.resize(bufferCapacity, 0.0f);
    state.cfStreamAccumulatedLeft = 0;
    state.cfStreamAccumulatedRight = 0;
    state.cfOutputLeft.clear();
    state.cfOutputRight.clear();
    state.cfOutputBufferLeft.clear();
    state.cfOutputBufferRight.clear();

    size_t outputCapacity = std::max(bufferCapacity, validOutputPerBlock);
    state.cfOutputLeft.reserve(outputCapacity);
    state.cfOutputRight.reserve(outputCapacity);
    state.cfOutputBufferLeft.reserve(outputCapacity);
    state.cfOutputBufferRight.reserve(outputCapacity);

    if (logCapacity) {
        std::cout << "  Crossfeed buffer capacity: " << bufferCapacity << " samples" << '\n';
    }
}

ConvolutionEngine::RateFamily resolveRateFamily(int inputSampleRate) {
    ConvolutionEngine::RateFamily family = ConvolutionEngine::detectRateFamily(inputSampleRate);
    if (family == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
        family = ConvolutionEngine::RateFamily::RATE_44K;
    }
    return family;
}

}  // namespace

CrossfeedInitResult initializeCrossfeed(daemon_app::RuntimeState& state,
                                        bool partitionedConvolutionEnabled) {
    CrossfeedInitResult result{};

    if (partitionedConvolutionEnabled) {
        std::cout << "[Partition] Crossfeed initialization skipped (low-latency mode)" << '\n';
        result.skipped = true;
        return result;
    }

    std::string hrtfDir = "data/crossfeed/hrtf";
    if (!std::filesystem::exists(hrtfDir)) {
        std::cout << "HRTF directory not found (" << hrtfDir << "), crossfeed feature disabled"
                  << '\n';
        std::cout << "  Hint: Run 'uv run python scripts/filters/generate_hrtf.py' to "
                     "generate HRTF "
                     "filters"
                  << '\n';
        return result;
    }

    std::cout << "Initializing HRTF processor for crossfeed..." << '\n';
    state.crossfeed.processor = new ConvolutionEngine::FourChannelFIR();

    ConvolutionEngine::RateFamily rateFamily = resolveRateFamily(state.rates.inputSampleRate);
    ConvolutionEngine::HeadSize initialHeadSize =
        ConvolutionEngine::stringToHeadSize(state.config.crossfeed.headSize);

    if (!state.crossfeed.processor->initialize(hrtfDir, state.config.blockSize, initialHeadSize,
                                               rateFamily)) {
        std::cerr << "  HRTF: Failed to initialize processor" << '\n';
        std::cerr << "  Hint: Run 'uv run python scripts/filters/generate_hrtf.py' to "
                     "generate HRTF "
                     "filters"
                  << '\n';
        delete state.crossfeed.processor;
        state.crossfeed.processor = nullptr;
        return result;
    }

    if (!state.crossfeed.processor->initializeStreaming()) {
        std::cerr << "  HRTF: Failed to initialize streaming mode" << '\n';
        delete state.crossfeed.processor;
        state.crossfeed.processor = nullptr;
        return result;
    }

    std::cout << "  HRTF processor ready (head size: "
              << ConvolutionEngine::headSizeToString(initialHeadSize) << ", rate family: "
              << (rateFamily == ConvolutionEngine::RateFamily::RATE_44K ? "44k" : "48k") << ")"
              << '\n';

    allocateCrossfeedBuffers(state.crossfeed, state.config,
                             state.crossfeed.processor->getStreamValidInputPerBlock(),
                             state.crossfeed.processor->getValidOutputPerBlock(), true);

    state.crossfeed.enabled.store(false);
    state.crossfeed.processor->setEnabled(false);
    std::cout << "  Crossfeed: initialized (disabled by default)" << '\n';

    result.initialized = true;
    return result;
}

void resetCrossfeedStreamStateLocked(daemon_app::CrossfeedState& state) {
    if (!state.cfStreamInputLeft.empty()) {
        std::fill(state.cfStreamInputLeft.begin(), state.cfStreamInputLeft.end(), 0.0f);
    }
    if (!state.cfStreamInputRight.empty()) {
        std::fill(state.cfStreamInputRight.begin(), state.cfStreamInputRight.end(), 0.0f);
    }
    state.cfStreamAccumulatedLeft = 0;
    state.cfStreamAccumulatedRight = 0;
    state.cfOutputLeft.clear();
    state.cfOutputRight.clear();
    if (state.processor) {
        state.processor->resetStreaming();
    }
}

void clearCrossfeedRuntimeBuffers(daemon_app::CrossfeedState& state) {
    state.cfStreamInputLeft.clear();
    state.cfStreamInputRight.clear();
    state.cfStreamAccumulatedLeft = 0;
    state.cfStreamAccumulatedRight = 0;
    state.cfOutputBufferLeft.clear();
    state.cfOutputBufferRight.clear();
}

CrossfeedSwitchStatus switchCrossfeedRateFamilyLocked(daemon_app::CrossfeedState& state,
                                                      const AppConfig& config,
                                                      ConvolutionEngine::RateFamily targetFamily) {
    if (!state.processor) {
        return CrossfeedSwitchStatus::NotInitialized;
    }

    if (!state.processor->switchRateFamily(targetFamily)) {
        return CrossfeedSwitchStatus::Failed;
    }

    resetCrossfeedStreamStateLocked(state);

    allocateCrossfeedBuffers(state, config, state.processor->getStreamValidInputPerBlock(),
                             state.processor->getValidOutputPerBlock(), false);

    return CrossfeedSwitchStatus::Switched;
}

void shutdownCrossfeed(daemon_app::CrossfeedState& state) {
    delete state.processor;
    state.processor = nullptr;
    state.enabled.store(false);
    state.cfStreamInputLeft.clear();
    state.cfStreamInputRight.clear();
    state.cfOutputLeft.clear();
    state.cfOutputRight.clear();
    state.cfOutputBufferLeft.clear();
    state.cfOutputBufferRight.clear();
    state.cfStreamAccumulatedLeft = 0;
    state.cfStreamAccumulatedRight = 0;
}

}  // namespace daemon_audio
