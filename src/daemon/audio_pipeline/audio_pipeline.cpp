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

    // 典型的には nFrames は ALSA/I2S の周期で固定。RT パスでの再確保を避けるため、
    // 初期化時点で十分な容量を確保しておく (Issue #894)。
    size_t initialFrames = static_cast<size_t>(DaemonConstants::DEFAULT_BLOCK_SIZE);
    if (deps_.config) {
        if (deps_.config->blockSize > 0) {
            initialFrames = std::max(initialFrames, static_cast<size_t>(deps_.config->blockSize));
        }
        if (deps_.config->periodSize > 0) {
            initialFrames = std::max(initialFrames, static_cast<size_t>(deps_.config->periodSize));
        }
        if (deps_.config->loopback.periodFrames > 0) {
            initialFrames =
                std::max(initialFrames, static_cast<size_t>(deps_.config->loopback.periodFrames));
        }
    }
    workLeft_.reserve(initialFrames);
    workRight_.reserve(initialFrames);
    workLeft_.resize(initialFrames);
    workRight_.resize(initialFrames);
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

    auto logLockSkipped = [&](const char* name, std::chrono::steady_clock::time_point& lastWarn) {
        auto now = std::chrono::steady_clock::now();
        if (now - lastWarn > std::chrono::seconds(5)) {
            LOG_WARN("AudioPipeline::process skipping block: {} mutex busy", name);
            lastWarn = now;
        }
    };

    if (deps_.streamingCacheManager) {
        deps_.streamingCacheManager->handleInputBlock();
    }

    std::unique_lock<std::mutex> inputLock(*deps_.inputMutex, std::try_to_lock);
    if (!inputLock.owns_lock()) {
        logLockSkipped("input", lastInputLockWarn_);

        // RT で待たない: ロックが取れない場合でもタイムラインを維持するため無音を出力。
        if (!deps_.config || !deps_.upsamplerOutputLeft || !deps_.upsamplerOutputRight) {
            return false;
        }
        auto ratio = static_cast<size_t>(deps_.config->upsampleRatio);
        size_t outputFrames = static_cast<size_t>(nFrames) * ratio;
        if (ratio == 0 || outputFrames == 0) {
            return false;
        }
        deps_.upsamplerOutputLeft->assign(outputFrames, 0.0f);
        deps_.upsamplerOutputRight->assign(outputFrames, 0.0f);

        size_t stored =
            enqueueOutputFramesLocked(*deps_.upsamplerOutputLeft, *deps_.upsamplerOutputRight);
        if (stored > 0 && deps_.buffer.playbackBuffer) {
            deps_.buffer.playbackBuffer->cv().notify_one();
        }
        return true;
    }

    const size_t frames = static_cast<size_t>(nFrames);
    if (frames > workLeft_.capacity() || frames > workRight_.capacity()) {
        LOG_ERROR("Input frames {} exceed work buffer capacity (L:{} R:{})", frames,
                  workLeft_.capacity(), workRight_.capacity());
        return false;
    }
    workLeft_.resize(frames);
    workRight_.resize(frames);
    AudioUtils::deinterleaveStereo(inputSamples, workLeft_.data(), workRight_.data(), nFrames);
    float inputPeak = computeStereoPeak(workLeft_.data(), workRight_.data(), frames);
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
        auto ratio = static_cast<size_t>(deps_.config->upsampleRatio);
        size_t outputFrames = frames * ratio;
        if (outputFrames > outputLeft->capacity() || outputFrames > outputRight->capacity()) {
            LOG_ERROR("Fallback output frames {} exceed buffer capacity (L:{} R:{})", outputFrames,
                      outputLeft->capacity(), outputRight->capacity());
            return false;
        }
        outputLeft->assign(outputFrames, 0.0f);
        outputRight->assign(outputFrames, 0.0f);
        for (size_t i = 0; i < frames; ++i) {
            size_t index = i * ratio;
            if (index < outputLeft->size()) {
                (*outputLeft)[index] = workLeft_[i];
                (*outputRight)[index] = workRight_[i];
            }
        }
        leftGenerated = true;
        rightGenerated = true;
    } else {
        if (!deps_.upsampler.process) {
            return false;
        }
        leftGenerated = deps_.upsampler.process(workLeft_.data(), nFrames, *outputLeft,
                                                deps_.upsampler.streamLeft, *deps_.streamInputLeft,
                                                *deps_.streamAccumulatedLeft);
        rightGenerated = deps_.upsampler.process(
            workRight_.data(), nFrames, *outputRight, deps_.upsampler.streamRight,
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
        std::unique_lock<std::mutex> cfLock(*deps_.crossfeedMutex, std::try_to_lock);
        if (cfLock.owns_lock()) {
            bool cfGenerated = deps_.crossfeedProcessor->processStreamBlock(
                outputLeft->data(), outputRight->data(), outputLeft->size(), *deps_.cfOutputLeft,
                *deps_.cfOutputRight, nullptr, *deps_.cfStreamInputLeft, *deps_.cfStreamInputRight,
                *deps_.cfStreamAccumulatedLeft, *deps_.cfStreamAccumulatedRight);
            if (cfGenerated) {
                size_t cfFrames = std::min(deps_.cfOutputLeft->size(), deps_.cfOutputRight->size());
                if (cfFrames > 0) {
                    float cfPeak = computeStereoPeak(deps_.cfOutputLeft->data(),
                                                     deps_.cfOutputRight->data(), cfFrames);
                    runtime_stats::updatePostCrossfeedPeak(cfPeak);
                }
                size_t stored =
                    enqueueOutputFramesLocked(*deps_.cfOutputLeft, *deps_.cfOutputRight);
                if (stored > 0 && deps_.buffer.playbackBuffer) {
                    deps_.buffer.playbackBuffer->cv().notify_one();
                }
                return true;
            }

            // クロスフィード有効時はストリーミングバッファへ蓄積だけ行い、
            // 十分なデータが溜まるまで元のアップサンプル出力をキューに積まない。
            // これにより未処理音声とクロスフィード済み音声が混在してバッファが膨張し
            // クラッシュするのを防ぐ。
            return true;
        }

        // ロック取得に失敗した場合はブロックせずアップサンプル出力をそのまま進める
        logLockSkipped("crossfeed", lastCrossfeedLockWarn_);
    }

    size_t stored = enqueueOutputFramesLocked(*outputLeft, *outputRight);
    if (stored > 0 && deps_.buffer.playbackBuffer) {
        deps_.buffer.playbackBuffer->cv().notify_one();
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
    if (frames == 0 || !hasBufferState()) {
        return result;
    }

    interleavedOut.resize(frames * 2);
    // Use floatScratch layout: [0..frames) left, [frames..2*frames) right,
    // [2*frames..4*frames) interleaved
    floatScratch.assign(frames * 4, 0.0f);
    float* planarLeft = floatScratch.data();
    float* planarRight = floatScratch.data() + frames;
    float* interleavedPtr = floatScratch.data() + (frames * 2);

    const float baseGain =
        (deps_.output.outputGain) ? deps_.output.outputGain->load(std::memory_order_relaxed) : 1.0f;

    bool hasAudio = deps_.buffer.playbackBuffer &&
                    deps_.buffer.playbackBuffer->readPlanar(planarLeft, planarRight, frames);

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

    // Interleave with gain before limiter/soft-mute
    AudioUtils::interleaveStereoWithGain(planarLeft, planarRight, interleavedPtr, frames, baseGain);

    if (softMute) {
        softMute->process(interleavedPtr, frames);
        using namespace DaemonConstants;
        if (softMute->getState() == SoftMute::MuteState::PLAYING &&
            softMute->getFadeDuration() > DEFAULT_SOFT_MUTE_FADE_MS) {
            softMute->setFadeDuration(DEFAULT_SOFT_MUTE_FADE_MS);
        }
    }

    float postGainPeak = applyOutputLimiter(interleavedPtr, frames);
    runtime_stats::updatePostGainPeak(postGainPeak);

    constexpr float kInt32MaxFloat = 2147483647.0f;
    for (size_t i = 0; i < frames; ++i) {
        float leftSample = interleavedPtr[i * 2];
        float rightSample = interleavedPtr[i * 2 + 1];

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
    (void)minFramesToRemove;
}

const BufferResources& AudioPipeline::bufferResources() const {
    return deps_.buffer;
}

bool AudioPipeline::hasBufferState() const {
    return deps_.buffer.playbackBuffer != nullptr;
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
    (void)minFramesToRemove;
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
