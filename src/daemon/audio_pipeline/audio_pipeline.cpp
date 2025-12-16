#include "daemon/audio_pipeline/audio_pipeline.h"

#include "audio/audio_utils.h"
#include "daemon/metrics/runtime_stats.h"
#include "logging/metrics.h"

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
        // Streaming mode may legitimately return false when it needs to accumulate more samples
        // before generating an output block. Treat "false + empty output" as "need more",
        // and only fail hard when it returns false but still produced output (unexpected) or
        // when one channel produced output while the other did not.
        const bool leftEmpty = (outputLeft->empty());
        const bool rightEmpty = (outputRight->empty());

        if (!leftGenerated && leftEmpty) {
            runtime_stats::recordUpsamplerNeedMoreBlock(true);
        } else if (!leftGenerated) {
            runtime_stats::recordUpsamplerErrorBlock(true);
        }

        if (!rightGenerated && rightEmpty) {
            runtime_stats::recordUpsamplerNeedMoreBlock(false);
        } else if (!rightGenerated) {
            runtime_stats::recordUpsamplerErrorBlock(false);
        }

        // If neither channel produced output yet, keep accumulating input without treating it
        // as a pipeline error.
        if (leftEmpty && rightEmpty) {
            return true;
        }

        return false;
    }

    size_t framesGenerated = std::min(outputLeft->size(), outputRight->size());
    if (framesGenerated > 0) {
        float upsamplerPeak =
            computeStereoPeak(outputLeft->data(), outputRight->data(), framesGenerated);
        runtime_stats::updateUpsamplerPeak(upsamplerPeak);
    }

    const bool crossfeedActive =
        deps_.crossfeedEnabled && deps_.crossfeedEnabled->load(std::memory_order_relaxed);

    if (crossfeedActive && deps_.crossfeedProcessor && deps_.crossfeedMutex && deps_.cfOutputLeft &&
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

        // クロスフィード有効時はストリーミングバッファへ蓄積だけ行い、
        // 十分なデータが溜まるまで元のアップサンプル出力をキューに積まない。
        // これにより未処理音声とクロスフィード済み音声が混在してバッファが膨張し
        // クラッシュするのを防ぐ。
        return true;
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

RenderResult AudioPipeline::renderOutput(size_t frames, std::vector<int32_t>& interleavedOut,
                                         std::vector<float>& floatScratch,
                                         SoftMute::Controller* softMute) {
    RenderResult result;
    result.framesRequested = frames;
    if (frames == 0 || !hasBufferState() || !deps_.buffer.bufferMutex) {
        return result;
    }

    interleavedOut.resize(frames * 2);
    floatScratch.assign(frames * 2, 0.0f);

    const float baseGain =
        (deps_.output.outputGain) ? deps_.output.outputGain->load(std::memory_order_relaxed) : 1.0f;

    bool hasAudio = false;
    {
        std::lock_guard<std::mutex> lock(*deps_.buffer.bufferMutex);
        size_t readable = deps_.buffer.outputBufferLeft->size();
        size_t readPos = *deps_.buffer.outputReadPos;
        if (readable >= readPos + frames) {
            hasAudio = true;
            floatScratch.resize(frames * 2);
            AudioUtils::interleaveStereoWithGain(deps_.buffer.outputBufferLeft->data() + readPos,
                                                 deps_.buffer.outputBufferRight->data() + readPos,
                                                 floatScratch.data(), frames, baseGain);
            *deps_.buffer.outputReadPos += frames;
            size_t cleanupFrames = std::max(frames * 4, static_cast<size_t>(1));
            trimInternal(cleanupFrames);
        }
    }

    if (!hasAudio) {
        // Output buffer underflow at daemon level (not necessarily ALSA XRUN).
        // This is a strong candidate for "ジッ/プチ" artifacts due to discontinuity.
        runtime_stats::recordRenderedSilenceBlock();
        runtime_stats::addRenderedSilenceFrames(frames);
        gpu_upsampler::metrics::recordBufferUnderflow();
        if (softMute && softMute->isTransitioning()) {
            softMute->process(floatScratch.data(), frames);
        }
        std::fill(interleavedOut.begin(), interleavedOut.end(), 0);
        result.framesRendered = frames;
        result.wroteSilence = true;
        return result;
    }

    if (softMute) {
        softMute->process(floatScratch.data(), frames);
        using namespace DaemonConstants;
        if (softMute->getState() == SoftMute::MuteState::PLAYING &&
            softMute->getFadeDuration() > DEFAULT_SOFT_MUTE_FADE_MS) {
            softMute->setFadeDuration(DEFAULT_SOFT_MUTE_FADE_MS);
        }
    }

    float postGainPeak = applyOutputLimiter(floatScratch.data(), frames);
    runtime_stats::updatePostGainPeak(postGainPeak);

    constexpr float kInt32MaxFloat = 2147483647.0f;
    for (size_t i = 0; i < frames; ++i) {
        float leftSample = floatScratch[i * 2];
        float rightSample = floatScratch[i * 2 + 1];

        if (leftSample > 1.0f || leftSample < -1.0f || rightSample > 1.0f || rightSample < -1.0f) {
            runtime_stats::recordClip();
        }

        leftSample = std::clamp(leftSample, -1.0f, 1.0f);
        rightSample = std::clamp(rightSample, -1.0f, 1.0f);

        interleavedOut[i * 2] = static_cast<int32_t>(std::lroundf(leftSample * kInt32MaxFloat));
        interleavedOut[i * 2 + 1] =
            static_cast<int32_t>(std::lroundf(rightSample * kInt32MaxFloat));
    }

    if (frames > 0) {
        runtime_stats::addSamples(frames * 2);
    }

    result.framesRendered = frames;
    result.wroteSilence = false;
    return result;
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

float AudioPipeline::applyOutputLimiter(float* interleaved, size_t frames) {
    constexpr float kEpsilon = 1e-6f;
    if (!interleaved || frames == 0) {
        if (deps_.output.limiterGain) {
            deps_.output.limiterGain->store(1.0f, std::memory_order_relaxed);
        }
        if (deps_.output.effectiveGain) {
            float base = (deps_.output.outputGain)
                             ? deps_.output.outputGain->load(std::memory_order_relaxed)
                             : 1.0f;
            deps_.output.effectiveGain->store(base, std::memory_order_relaxed);
        }
        return 0.0f;
    }

    float peak = AudioUtils::computeInterleavedPeak(interleaved, frames);
    float limiterGain = 1.0f;
    float target = deps_.config ? deps_.config->headroomTarget : 0.0f;
    if (target > 0.0f && peak > target) {
        limiterGain = target / (peak + kEpsilon);
        AudioUtils::applyInterleavedGain(interleaved, frames, limiterGain);
        peak = target;
    }

    if (deps_.output.limiterGain) {
        deps_.output.limiterGain->store(limiterGain, std::memory_order_relaxed);
    }
    if (deps_.output.effectiveGain) {
        float base = (deps_.output.outputGain)
                         ? deps_.output.outputGain->load(std::memory_order_relaxed)
                         : 1.0f;
        deps_.output.effectiveGain->store(base * limiterGain, std::memory_order_relaxed);
    }

    return peak;
}

}  // namespace audio_pipeline
