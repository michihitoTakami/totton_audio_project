#include "daemon/audio_pipeline/audio_pipeline.h"

#include "audio_utils.h"
#include "daemon/metrics/runtime_stats.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace audio_pipeline {

AudioPipeline::AudioPipeline(Dependencies deps) : deps_(std::move(deps)) {
    if (!deps_.maxOutputBufferFrames) {
        deps_.maxOutputBufferFrames = []() { return static_cast<size_t>(1); };
    }
    if (!deps_.currentOutputRate) {
        deps_.currentOutputRate = []() { return DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE; };
    }
}

bool AudioPipeline::process(const float* inputSamples, uint32_t nFrames) {
    if (!inputSamples || nFrames == 0 || !isUpsamplerAvailable() || !hasBufferState() ||
        !deps_.inputMutex || !deps_.streamInputLeft || !deps_.streamInputRight ||
        !deps_.streamAccumulatedLeft || !deps_.streamAccumulatedRight ||
        !deps_.upsamplerOutputLeft || !deps_.upsamplerOutputRight) {
        return false;
    }

    if (!isOutputReady()) {
        logDroppingInput();
        return false;
    }

    if (deps_.streamingCacheManager) {
        deps_.streamingCacheManager->handleInputBlock();
    }

    std::lock_guard<std::mutex> inputLock(*deps_.inputMutex);

    std::vector<float> left(nFrames);
    std::vector<float> right(nFrames);
    AudioUtils::deinterleaveStereo(inputSamples, left.data(), right.data(), nFrames);
    float inputPeak = computeStereoPeak(left.data(), right.data(), nFrames);
    runtime_stats::updateInputPeak(inputPeak);

    auto* outputLeft = deps_.upsamplerOutputLeft;
    auto* outputRight = deps_.upsamplerOutputRight;

    bool useFallback =
        deps_.fallbackActive && deps_.fallbackActive->load(std::memory_order_relaxed);
    bool leftGenerated = false;
    bool rightGenerated = false;

    if (useFallback) {
        if (!deps_.config) {
            return false;
        }
        size_t ratio = static_cast<size_t>(deps_.config->upsampleRatio);
        size_t outputFrames = static_cast<size_t>(nFrames) * ratio;
        outputLeft->assign(outputFrames, 0.0f);
        outputRight->assign(outputFrames, 0.0f);
        for (size_t i = 0; i < nFrames; ++i) {
            size_t index = i * ratio;
            if (index < outputLeft->size()) {
                (*outputLeft)[index] = left[i];
                (*outputRight)[index] = right[i];
            }
        }
        leftGenerated = true;
        rightGenerated = true;
    } else {
        if (!deps_.upsampler.process) {
            return false;
        }
        leftGenerated =
            deps_.upsampler.process(left.data(), nFrames, *outputLeft, deps_.upsampler.streamLeft,
                                    *deps_.streamInputLeft, *deps_.streamAccumulatedLeft);
        rightGenerated = deps_.upsampler.process(
            right.data(), nFrames, *outputRight, deps_.upsampler.streamRight,
            *deps_.streamInputRight, *deps_.streamAccumulatedRight);
    }

    if (!leftGenerated || !rightGenerated) {
        return false;
    }

    size_t framesGenerated = std::min(outputLeft->size(), outputRight->size());
    if (framesGenerated > 0) {
        float upsamplerPeak =
            computeStereoPeak(outputLeft->data(), outputRight->data(), framesGenerated);
        runtime_stats::updateUpsamplerPeak(upsamplerPeak);
    }

    if (deps_.crossfeedEnabled && deps_.crossfeedEnabled->load(std::memory_order_relaxed) &&
        deps_.crossfeedProcessor && deps_.crossfeedMutex && deps_.cfOutputLeft &&
        deps_.cfOutputRight && deps_.cfStreamInputLeft && deps_.cfStreamInputRight &&
        deps_.cfStreamAccumulatedLeft && deps_.cfStreamAccumulatedRight) {
        std::lock_guard<std::mutex> cfLock(*deps_.crossfeedMutex);
        bool cfGenerated = deps_.crossfeedProcessor->processStreamBlock(
            outputLeft->data(), outputRight->data(), outputLeft->size(), *deps_.cfOutputLeft,
            *deps_.cfOutputRight, 0, *deps_.cfStreamInputLeft, *deps_.cfStreamInputRight,
            *deps_.cfStreamAccumulatedLeft, *deps_.cfStreamAccumulatedRight);
        if (cfGenerated) {
            size_t cfFrames = std::min(deps_.cfOutputLeft->size(), deps_.cfOutputRight->size());
            if (cfFrames > 0) {
                float cfPeak = computeStereoPeak(deps_.cfOutputLeft->data(),
                                                 deps_.cfOutputRight->data(), cfFrames);
                runtime_stats::updatePostCrossfeedPeak(cfPeak);
            }
            if (deps_.buffer.bufferMutex) {
                std::lock_guard<std::mutex> lock(*deps_.buffer.bufferMutex);
                enqueueOutputFramesLocked(*deps_.cfOutputLeft, *deps_.cfOutputRight);
            }
            if (deps_.buffer.bufferCv) {
                deps_.buffer.bufferCv->notify_one();
            }
            return true;
        }
    }

    if (deps_.buffer.bufferMutex) {
        std::lock_guard<std::mutex> lock(*deps_.buffer.bufferMutex);
        enqueueOutputFramesLocked(*outputLeft, *outputRight);
    }
    if (deps_.buffer.bufferCv) {
        deps_.buffer.bufferCv->notify_one();
    }

    if (framesGenerated > 0) {
        float postPeak =
            computeStereoPeak(outputLeft->data(), outputRight->data(), framesGenerated);
        runtime_stats::updatePostCrossfeedPeak(postPeak);
    }

    return true;
}

void AudioPipeline::trimOutputBuffer(size_t minFramesToRemove) {
    if (!hasBufferState() || !deps_.buffer.bufferMutex) {
        return;
    }
    std::lock_guard<std::mutex> lock(*deps_.buffer.bufferMutex);
    trimInternal(minFramesToRemove);
}

const BufferResources& AudioPipeline::bufferResources() const {
    return deps_.buffer;
}

bool AudioPipeline::hasBufferState() const {
    return deps_.buffer.outputBufferLeft && deps_.buffer.outputBufferRight &&
           deps_.buffer.outputReadPos && deps_.buffer.bufferMutex;
}

bool AudioPipeline::isUpsamplerAvailable() const {
    return deps_.upsampler.available;
}

bool AudioPipeline::isOutputReady() const {
    return deps_.outputReady && deps_.outputReady->load(std::memory_order_relaxed);
}

void AudioPipeline::logDroppingInput() {
    auto now = std::chrono::steady_clock::now();
    if (now - lastDropWarn_ > std::chrono::seconds(5)) {
        LOG_DEBUG("Dropping input: DAC not ready");
        lastDropWarn_ = now;
    }
}

void AudioPipeline::trimInternal(size_t minFramesToRemove) {
    if (!hasBufferState()) {
        return;
    }

    auto& buffers = deps_.buffer;
    size_t readable = buffers.outputBufferLeft->size();
    if (readable == 0) {
        *buffers.outputReadPos = 0;
        return;
    }

    if (*buffers.outputReadPos == 0) {
        return;
    }

    if (*buffers.outputReadPos > readable) {
        *buffers.outputReadPos = readable;
    }

    if ((minFramesToRemove > 0 && *buffers.outputReadPos >= minFramesToRemove) ||
        *buffers.outputReadPos == readable) {
        auto eraseCount = static_cast<std::ptrdiff_t>(*buffers.outputReadPos);
        buffers.outputBufferLeft->erase(buffers.outputBufferLeft->begin(),
                                        buffers.outputBufferLeft->begin() + eraseCount);
        buffers.outputBufferRight->erase(buffers.outputBufferRight->begin(),
                                         buffers.outputBufferRight->begin() + eraseCount);
        *buffers.outputReadPos = 0;
    }
}

float AudioPipeline::computeStereoPeak(const float* left, const float* right, size_t frames) const {
    if (!left || !right || frames == 0) {
        return 0.0f;
    }
    float peak = 0.0f;
    for (size_t i = 0; i < frames; ++i) {
        float l = std::fabs(left[i]);
        float r = std::fabs(right[i]);
        peak = std::max(peak, std::max(l, r));
    }
    return peak;
}

}  // namespace audio_pipeline
