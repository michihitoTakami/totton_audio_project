#include "daemon/audio_pipeline/audio_pipeline.h"

#include "audio/audio_utils.h"
#include "audio/overlap_add.h"
#include "daemon/metrics/runtime_stats.h"
#include "delimiter/inference_backend.h"
#include "logging/metrics.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <thread>
#include <vector>

namespace audio_pipeline {

namespace {

constexpr float kDefaultChunkSec = 6.0f;
constexpr float kDefaultOverlapSec = 0.25f;
constexpr int kMaxHighLatencyInitRate = 192000;
constexpr float kHighLatencyInputBufferSeconds = 20.0f;
constexpr std::size_t kDelimiterFailureCountToBypass = 3;
constexpr std::size_t kDelimiterOverloadCountToBypass = 3;
constexpr std::size_t kDelimiterRecoveryCountToRestore = 5;
constexpr double kDelimiterMaxRealtimeFactor = 1.15;
constexpr double kDelimiterMaxQueueSecondsFactor = 2.0;

inline size_t safeRoundFrames(double seconds, int sampleRate) {
    if (seconds <= 0.0 || sampleRate <= 0) {
        return 0;
    }
    double frames = seconds * static_cast<double>(sampleRate);
    if (frames < 1.0) {
        return 1;
    }
    return static_cast<size_t>(std::llround(frames));
}

inline int pickInitialInputRate(const Dependencies& deps) {
    if (deps.currentInputRate) {
        int rate = deps.currentInputRate();
        if (rate > 0) {
            return rate;
        }
    }
    if (deps.config && deps.config->loopback.sampleRate > 0) {
        return static_cast<int>(deps.config->loopback.sampleRate);
    }
    if (deps.config && deps.config->i2s.sampleRate > 0) {
        return static_cast<int>(deps.config->i2s.sampleRate);
    }
    if (deps.config && deps.config->delimiter.expectedSampleRate > 0) {
        return static_cast<int>(deps.config->delimiter.expectedSampleRate);
    }
    return DaemonConstants::DEFAULT_INPUT_SAMPLE_RATE;
}

}  // namespace

AudioPipeline::AudioPipeline(Dependencies deps) : deps_(std::move(deps)) {
    if (!deps_.maxOutputBufferFrames) {
        deps_.maxOutputBufferFrames = []() { return static_cast<size_t>(1); };
    }
    if (!deps_.currentOutputRate) {
        deps_.currentOutputRate = []() { return DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE; };
    }
    if (!deps_.currentInputRate) {
        deps_.currentInputRate = []() { return DaemonConstants::DEFAULT_INPUT_SAMPLE_RATE; };
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

    highLatencyEnabled_ = (deps_.config && deps_.config->delimiter.enabled);
    if (highLatencyEnabled_) {
        if (deps_.delimiterMode) {
            deps_.delimiterMode->store(static_cast<int>(delimiter::ProcessingMode::Active),
                                       std::memory_order_relaxed);
        }
        if (deps_.delimiterFallbackReason) {
            deps_.delimiterFallbackReason->store(static_cast<int>(delimiter::FallbackReason::None),
                                                 std::memory_order_relaxed);
        }
        if (deps_.delimiterBypassLocked) {
            deps_.delimiterBypassLocked->store(false, std::memory_order_relaxed);
        }
    }
    if (highLatencyEnabled_) {
        if (deps_.delimiterBackendFactory) {
            delimiterBackend_ = deps_.delimiterBackendFactory(deps_.config->delimiter);
        } else {
            delimiterBackend_ = delimiter::createDelimiterInferenceBackend(deps_.config->delimiter);
        }

        int initRate = pickInitialInputRate(deps_);
        int rateForCapacity = std::clamp(initRate, 1, kMaxHighLatencyInitRate);
        size_t capacityFrames =
            safeRoundFrames(static_cast<double>(kHighLatencyInputBufferSeconds), rateForCapacity);
        capacityFrames = std::max<size_t>(capacityFrames, 1);
        inputInterleaved_.init(capacityFrames * 2);

        workerStop_.store(false, std::memory_order_release);
        workerFailed_.store(false, std::memory_order_release);
        workerThread_ = std::thread([this]() { workerLoop(); });

        LOG_INFO("[AudioPipeline] High-latency worker enabled (delimiter backend: {})",
                 delimiterBackend_ ? delimiterBackend_->name() : "null");
    }
}

AudioPipeline::~AudioPipeline() {
    workerStop_.store(true, std::memory_order_release);
    inputCv_.notify_all();
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
}

bool AudioPipeline::process(const float* inputSamples, uint32_t nFrames) {
    if (!highLatencyEnabled_) {
        return processDirect(inputSamples, nFrames);
    }
    return enqueueInputForWorker(inputSamples, nFrames);
}

bool AudioPipeline::processDirect(const float* inputSamples, uint32_t nFrames) {
    struct RtScopeGuard {
        std::atomic<bool>& flag;
        explicit RtScopeGuard(std::atomic<bool>& f) : flag(f) {
            flag.store(true, std::memory_order_release);
        }
        ~RtScopeGuard() {
            flag.store(false, std::memory_order_release);
        }
    } rtScope(rtInProcess_);

    if (!inputSamples || nFrames == 0 || !isUpsamplerAvailable() || !hasBufferState() ||
        !deps_.streamInputLeft || !deps_.streamInputRight || !deps_.streamAccumulatedLeft ||
        !deps_.streamAccumulatedRight || !deps_.upsamplerOutputLeft ||
        !deps_.upsamplerOutputRight) {
        return false;
    }

    const bool pauseRequested = (pauseRequestCount_.load(std::memory_order_acquire) > 0);
    rtPaused_.store(pauseRequested, std::memory_order_release);

    const bool cacheFlushPending =
        deps_.streamingCacheManager && deps_.streamingCacheManager->hasPendingFlush();

    if (!pauseRequested && deps_.streamingCacheManager) {
        deps_.streamingCacheManager->handleInputBlock();
    }

    if (!isOutputReady()) {
        logDroppingInput();
        return false;
    }

    if (pauseRequested || cacheFlushPending) {
        // RT 側はブロッキングせず、要求中は無音を供給してタイムラインを維持する。
        if (!deps_.config || !deps_.upsamplerOutputLeft || !deps_.upsamplerOutputRight) {
            return false;
        }
        size_t ratio = static_cast<size_t>(deps_.config->upsampleRatio);
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
    const bool crossfeedReset =
        deps_.crossfeedResetRequested &&
        deps_.crossfeedResetRequested->exchange(false, std::memory_order_acq_rel);

    if (deps_.crossfeedProcessor && deps_.cfOutputLeft && deps_.cfOutputRight &&
        deps_.cfStreamInputLeft && deps_.cfStreamInputRight && deps_.cfStreamAccumulatedLeft &&
        deps_.cfStreamAccumulatedRight) {
        if ((crossfeedActive != lastCrossfeedEnabledApplied_) || crossfeedReset) {
            deps_.crossfeedProcessor->setEnabled(crossfeedActive);
            if (!deps_.cfStreamInputLeft->empty()) {
                std::fill(deps_.cfStreamInputLeft->begin(), deps_.cfStreamInputLeft->end(), 0.0f);
            }
            if (!deps_.cfStreamInputRight->empty()) {
                std::fill(deps_.cfStreamInputRight->begin(), deps_.cfStreamInputRight->end(), 0.0f);
            }
            *deps_.cfStreamAccumulatedLeft = 0;
            *deps_.cfStreamAccumulatedRight = 0;
            deps_.cfOutputLeft->clear();
            deps_.cfOutputRight->clear();
            deps_.crossfeedProcessor->resetStreaming();
            lastCrossfeedEnabledApplied_ = crossfeedActive;
        }
    }

    if (crossfeedActive && deps_.crossfeedProcessor && deps_.cfOutputLeft && deps_.cfOutputRight &&
        deps_.cfStreamInputLeft && deps_.cfStreamInputRight && deps_.cfStreamAccumulatedLeft &&
        deps_.cfStreamAccumulatedRight) {
        // NOTE: Upsampler の 1回の出力ブロックが大きい場合（例: 705.6kHz で 40万sample超）
        // FourChannelFIR のストリーミング入力バッファ（通常 2x block 分）に一気に詰め込めず、
        // "Input buffer too small" → 状態ドロップ → 音飛びにつながる。
        //
        // 対策: Crossfeed 側の streamValidInputPerBlock 境界に合わせて入力を分割し、
        // 1回の呼び出しで溜め込み過ぎない + 可能な限り同一 RT サイクル内で処理を進める。
        const size_t streamBlock = deps_.crossfeedProcessor->getStreamValidInputPerBlock();
        const size_t inputBufL = deps_.cfStreamInputLeft->size();
        const size_t inputBufR = deps_.cfStreamInputRight->size();
        const size_t totalFrames = framesGenerated;

        if (streamBlock == 0 || inputBufL < streamBlock || inputBufR < streamBlock) {
            LOG_EVERY_N(ERROR, 100,
                        "[AudioPipeline] Crossfeed stream buffers undersized (streamBlock={}, "
                        "bufL={}, bufR={}). Falling back to raw upsampler output for this block.",
                        streamBlock, inputBufL, inputBufR);
        } else {
            bool enqueuedAnyCrossfeedOutput = false;
            size_t offset = 0;
            while (offset < totalFrames) {
                size_t accL = *deps_.cfStreamAccumulatedLeft;
                size_t accR = *deps_.cfStreamAccumulatedRight;
                size_t acc = std::min(accL, accR);

                // Safety: these should stay in sync and < streamBlock; recover defensively if not.
                if (accL != accR || acc >= streamBlock) {
                    LOG_EVERY_N(ERROR, 100,
                                "[AudioPipeline] Crossfeed accumulated mismatch/overflow "
                                "(accL={}, accR={}, streamBlock={}), resetting streaming state",
                                accL, accR, streamBlock);
                    *deps_.cfStreamAccumulatedLeft = 0;
                    *deps_.cfStreamAccumulatedRight = 0;
                    deps_.cfOutputLeft->clear();
                    deps_.cfOutputRight->clear();
                    deps_.crossfeedProcessor->resetStreaming();
                    acc = 0;
                }

                const size_t remaining = totalFrames - offset;
                const size_t needToFill = streamBlock - acc;
                const size_t space = std::min(inputBufL - acc, inputBufR - acc);
                size_t chunk = std::min({remaining, needToFill, space});
                if (chunk == 0) {
                    // Should not happen (buffers are expected to be >= streamBlock*2).
                    // Recover by resetting accumulation and continue.
                    LOG_EVERY_N(ERROR, 100,
                                "[AudioPipeline] Crossfeed chunk became 0 (remaining={}, acc={}, "
                                "streamBlock={}, bufL={}, bufR={}), resetting accumulation",
                                remaining, acc, streamBlock, inputBufL, inputBufR);
                    *deps_.cfStreamAccumulatedLeft = 0;
                    *deps_.cfStreamAccumulatedRight = 0;
                    deps_.cfOutputLeft->clear();
                    deps_.cfOutputRight->clear();
                    deps_.crossfeedProcessor->resetStreaming();
                    continue;
                }

                bool cfGenerated = deps_.crossfeedProcessor->processStreamBlock(
                    outputLeft->data() + static_cast<std::ptrdiff_t>(offset),
                    outputRight->data() + static_cast<std::ptrdiff_t>(offset), chunk,
                    *deps_.cfOutputLeft, *deps_.cfOutputRight, nullptr, *deps_.cfStreamInputLeft,
                    *deps_.cfStreamInputRight, *deps_.cfStreamAccumulatedLeft,
                    *deps_.cfStreamAccumulatedRight);
                offset += chunk;

                if (cfGenerated) {
                    size_t cfFrames =
                        std::min(deps_.cfOutputLeft->size(), deps_.cfOutputRight->size());
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
                    enqueuedAnyCrossfeedOutput = enqueuedAnyCrossfeedOutput || (stored > 0);
                    deps_.cfOutputLeft->clear();
                    deps_.cfOutputRight->clear();
                }
            }

            // クロスフィード有効時はストリーミングバッファへ蓄積だけ行い、
            // 十分なデータが溜まるまで元のアップサンプル出力をキューに積まない。
            // これにより未処理音声とクロスフィード済み音声が混在してバッファが膨張し
            // クラッシュするのを防ぐ。
            //
            // ただし、有効化直後は 4ch FIR がまだ出力できず "何も enqueue しない" 状態が続くと
            // PlaybackBuffer が枯渇して underflow → クリック/音飛びの原因になる。
            // そのため、この RT サイクルで 1フレームも enqueue できなかった場合は、
            // タイムライン維持のため無音を enqueue する。
            if (!enqueuedAnyCrossfeedOutput && framesGenerated > 0) {
                std::fill(outputLeft->begin(), outputLeft->end(), 0.0f);
                std::fill(outputRight->begin(), outputRight->end(), 0.0f);
                size_t stored = enqueueOutputFramesLocked(*outputLeft, *outputRight);
                if (stored > 0 && deps_.buffer.playbackBuffer) {
                    deps_.buffer.playbackBuffer->cv().notify_one();
                }
            }
            return true;
        }
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

bool AudioPipeline::enqueueInputForWorker(const float* inputSamples, uint32_t nFrames) {
    if (!inputSamples || nFrames == 0 || !hasBufferState()) {
        return false;
    }

    if (workerFailed_.load(std::memory_order_acquire)) {
        // Safety-first: keep playback stable with silence when worker is down.
        if (!deps_.config || !deps_.upsamplerOutputLeft || !deps_.upsamplerOutputRight) {
            return false;
        }
        size_t ratio = static_cast<size_t>(deps_.config->upsampleRatio);
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

    const bool pauseRequested = (pauseRequestCount_.load(std::memory_order_acquire) > 0);
    rtPaused_.store(pauseRequested, std::memory_order_release);

    const bool cacheFlushPending =
        deps_.streamingCacheManager && deps_.streamingCacheManager->hasPendingFlush();

    if (!pauseRequested && deps_.streamingCacheManager) {
        deps_.streamingCacheManager->handleInputBlock();
    }

    if (!isOutputReady()) {
        logDroppingInput();
        return false;
    }

    if (pauseRequested || cacheFlushPending) {
        // Keep timeline stable and avoid mixing pre/post-switch audio.
        inputDropDetected_.store(true, std::memory_order_release);
        inputCv_.notify_one();
        if (!deps_.config || !deps_.upsamplerOutputLeft || !deps_.upsamplerOutputRight) {
            return false;
        }
        size_t ratio = static_cast<size_t>(deps_.config->upsampleRatio);
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

    const size_t samples = static_cast<size_t>(nFrames) * 2;
    if (samples == 0) {
        return false;
    }

    if (!inputInterleaved_.write(inputSamples, samples)) {
        inputDropDetected_.store(true, std::memory_order_release);
        logDroppingHighLatencyInput();
        return false;
    }

    inputCv_.notify_one();
    return true;
}

void AudioPipeline::requestRtPause() {
    pauseRequestCount_.fetch_add(1, std::memory_order_acq_rel);
}

void AudioPipeline::resumeRtPause() {
    int prev = pauseRequestCount_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 1) {
        pauseRequestCount_.store(0, std::memory_order_release);
    }
}

bool AudioPipeline::waitForRtPaused(std::chrono::milliseconds timeout) const {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        if (rtPaused_.load(std::memory_order_acquire)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return rtPaused_.load(std::memory_order_acquire);
}

bool AudioPipeline::waitForRtQuiescent(std::chrono::milliseconds timeout) const {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        if (rtPaused_.load(std::memory_order_acquire)) {
            return true;
        }
        if (!rtInProcess_.load(std::memory_order_acquire)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return rtPaused_.load(std::memory_order_acquire) ||
           !rtInProcess_.load(std::memory_order_acquire);
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

void AudioPipeline::logDroppingHighLatencyInput() {
    auto now = std::chrono::steady_clock::now();
    if (now - lastInputDropWarn_ > std::chrono::seconds(5)) {
        size_t available = inputInterleaved_.availableToRead();
        size_t capacity = inputInterleaved_.capacity();
        LOG_WARN(
            "[AudioPipeline] High-latency input queue overflow (available={} samples, cap={}), "
            "dropping input",
            available, capacity);
        lastInputDropWarn_ = now;
    }
}

void AudioPipeline::resetHighLatencyState(const char* reason) {
    std::lock_guard<std::mutex> lock(inputCvMutex_);
    resetHighLatencyStateLocked(reason);
}

void AudioPipeline::resetHighLatencyStateLocked(const char* reason) {
    (void)reason;
    hasPrevChunk_ = false;
    if (overlapFrames_ > 0) {
        prevInputTailLeft_.assign(overlapFrames_, 0.0f);
        prevInputTailRight_.assign(overlapFrames_, 0.0f);
        prevOutputTailWeightedLeft_.assign(overlapFrames_, 0.0f);
        prevOutputTailWeightedRight_.assign(overlapFrames_, 0.0f);
    } else {
        prevInputTailLeft_.clear();
        prevInputTailRight_.clear();
        prevOutputTailWeightedLeft_.clear();
        prevOutputTailWeightedRight_.clear();
    }
    chunkInputLeft_.clear();
    chunkInputRight_.clear();
    chunkOutputLeft_.clear();
    chunkOutputRight_.clear();
    segmentLeft_.clear();
    segmentRight_.clear();
    readInterleaved_.clear();
    readLeft_.clear();
    readRight_.clear();
    if (delimiterBackend_) {
        delimiterBackend_->reset();
    }
}

void AudioPipeline::updateDelimiterStatus(const delimiter::SafetyStatus& status) {
    if (deps_.delimiterMode) {
        deps_.delimiterMode->store(static_cast<int>(status.mode), std::memory_order_relaxed);
    }
    if (deps_.delimiterFallbackReason) {
        deps_.delimiterFallbackReason->store(static_cast<int>(status.lastFallbackReason),
                                             std::memory_order_relaxed);
    }
    if (deps_.delimiterBypassLocked) {
        deps_.delimiterBypassLocked->store(status.bypassLocked, std::memory_order_relaxed);
    }
}

void AudioPipeline::workerLoop() {
    try {
        int inputRate = pickInitialInputRate(deps_);
        if (deps_.currentInputRate) {
            int probed = deps_.currentInputRate();
            if (probed > 0) {
                inputRate = probed;
            }
        }
        workerInputRate_ = inputRate;

        float chunkSec = (deps_.config && deps_.config->delimiter.chunkSec > 0.0f)
                             ? deps_.config->delimiter.chunkSec
                             : kDefaultChunkSec;
        float overlapSec = (deps_.config && deps_.config->delimiter.overlapSec > 0.0f)
                               ? deps_.config->delimiter.overlapSec
                               : kDefaultOverlapSec;

        chunkFrames_ = safeRoundFrames(chunkSec, workerInputRate_);
        overlapFrames_ = safeRoundFrames(overlapSec, workerInputRate_);
        overlapFrames_ = std::min(overlapFrames_, chunkFrames_ > 0 ? (chunkFrames_ - 1) : 0);
        if (chunkFrames_ == 0 || overlapFrames_ == 0 || overlapFrames_ >= chunkFrames_) {
            LOG_ERROR(
                "[AudioPipeline] Invalid delimiter chunk parameters (chunkFrames={}, "
                "overlapFrames={}, rate={})",
                chunkFrames_, overlapFrames_, workerInputRate_);
            workerFailed_.store(true, std::memory_order_release);
            return;
        }
        if (overlapFrames_ > (chunkFrames_ / 2)) {
            LOG_WARN(
                "[AudioPipeline] overlapFrames too large (chunkFrames={}, overlapFrames={}), "
                "clamping to chunk/2",
                chunkFrames_, overlapFrames_);
            overlapFrames_ = chunkFrames_ / 2;
        }
        hopFrames_ = chunkFrames_ - overlapFrames_;
        fadeIn_ = AudioUtils::makeRaisedCosineFade(overlapFrames_);

        prevInputTailLeft_.assign(overlapFrames_, 0.0f);
        prevInputTailRight_.assign(overlapFrames_, 0.0f);
        prevOutputTailWeightedLeft_.assign(overlapFrames_, 0.0f);
        prevOutputTailWeightedRight_.assign(overlapFrames_, 0.0f);

        const size_t maxFramesPerCall = std::max<size_t>(1, workLeft_.capacity());
        downstreamInterleaved_.assign(maxFramesPerCall * 2, 0.0f);

        const double maxQueueSeconds =
            std::max(1.0, static_cast<double>(chunkSec) * kDelimiterMaxQueueSecondsFactor);
        delimiter::SafetyConfig safetyConfig;
        safetyConfig.sampleRate = workerInputRate_;
        safetyConfig.fadeDurationMs = std::max(1, static_cast<int>(std::lround(overlapSec * 1000)));
        safetyConfig.failureCountToBypass = kDelimiterFailureCountToBypass;
        safetyConfig.overloadCountToBypass = kDelimiterOverloadCountToBypass;
        safetyConfig.recoveryCountToRestore = kDelimiterRecoveryCountToRestore;
        safetyConfig.maxRealtimeFactor = kDelimiterMaxRealtimeFactor;
        safetyConfig.maxQueueSeconds = maxQueueSeconds;
        safetyConfig.lockOnFailure = true;
        safetyConfig.lockOnOverload = false;
        delimiterSafety_ = std::make_unique<delimiter::SafetyController>(safetyConfig);
        updateDelimiterStatus(delimiterSafety_->status());

        while (!workerStop_.load(std::memory_order_acquire)) {
            if (deps_.running && !deps_.running->load(std::memory_order_acquire)) {
                break;
            }

            if (inputDropDetected_.exchange(false, std::memory_order_acq_rel)) {
                resetHighLatencyState("input drop");

                // Drop all pending input to avoid stitching mismatched segments.
                size_t avail = inputInterleaved_.availableToRead();
                if (avail > 0) {
                    readInterleaved_.assign(std::min(avail, inputInterleaved_.capacity()), 0.0f);
                    while (inputInterleaved_.availableToRead() > 0) {
                        size_t chunk =
                            std::min(readInterleaved_.size(), inputInterleaved_.availableToRead());
                        (void)inputInterleaved_.read(readInterleaved_.data(), chunk);
                    }
                }
                if (delimiterSafety_) {
                    (void)delimiterSafety_->observeOverload(kDelimiterMaxRealtimeFactor + 1.0,
                                                            maxQueueSeconds + 1.0);
                    updateDelimiterStatus(delimiterSafety_->status());
                }
            }

            const bool needFullChunk = !hasPrevChunk_;
            const size_t needFrames = needFullChunk ? chunkFrames_ : hopFrames_;
            const size_t needSamples = needFrames * 2;

            {
                std::unique_lock<std::mutex> lock(inputCvMutex_);
                inputCv_.wait_for(lock, std::chrono::milliseconds(50), [&]() {
                    return workerStop_.load(std::memory_order_acquire) ||
                           (deps_.running && !deps_.running->load(std::memory_order_acquire)) ||
                           inputInterleaved_.availableToRead() >= needSamples;
                });
            }

            if (workerStop_.load(std::memory_order_acquire)) {
                break;
            }
            if (deps_.running && !deps_.running->load(std::memory_order_acquire)) {
                break;
            }
            if (inputInterleaved_.availableToRead() < needSamples) {
                continue;
            }

            readInterleaved_.assign(needSamples, 0.0f);
            if (!inputInterleaved_.read(readInterleaved_.data(), needSamples)) {
                continue;
            }

            readLeft_.assign(needFrames, 0.0f);
            readRight_.assign(needFrames, 0.0f);
            for (size_t i = 0; i < needFrames; ++i) {
                readLeft_[i] = readInterleaved_[i * 2];
                readRight_[i] = readInterleaved_[i * 2 + 1];
            }

            chunkInputLeft_.assign(chunkFrames_, 0.0f);
            chunkInputRight_.assign(chunkFrames_, 0.0f);
            if (needFullChunk) {
                std::copy(readLeft_.begin(), readLeft_.end(), chunkInputLeft_.begin());
                std::copy(readRight_.begin(), readRight_.end(), chunkInputRight_.begin());
            } else {
                std::copy(prevInputTailLeft_.begin(), prevInputTailLeft_.end(),
                          chunkInputLeft_.begin());
                std::copy(prevInputTailRight_.begin(), prevInputTailRight_.end(),
                          chunkInputRight_.begin());
                std::copy(readLeft_.begin(), readLeft_.end(),
                          chunkInputLeft_.begin() + static_cast<std::ptrdiff_t>(overlapFrames_));
                std::copy(readRight_.begin(), readRight_.end(),
                          chunkInputRight_.begin() + static_cast<std::ptrdiff_t>(overlapFrames_));
            }

            std::copy(chunkInputLeft_.end() - static_cast<std::ptrdiff_t>(overlapFrames_),
                      chunkInputLeft_.end(), prevInputTailLeft_.begin());
            std::copy(chunkInputRight_.end() - static_cast<std::ptrdiff_t>(overlapFrames_),
                      chunkInputRight_.end(), prevInputTailRight_.begin());

            bool inferenceOk = false;
            bool inferenceAttempted = false;
            bool bypassChunk = false;
            double realtimeFactor = 0.0;

            if (delimiterSafety_) {
                auto status = delimiterSafety_->status();
                bypassChunk =
                    (status.mode == delimiter::ProcessingMode::Bypass) || status.bypassLocked;
            }

            if (!bypassChunk && delimiterBackend_ && deps_.config &&
                deps_.config->delimiter.enabled &&
                static_cast<int>(delimiterBackend_->expectedSampleRate()) == workerInputRate_) {
                inferenceAttempted = true;
                auto start = std::chrono::steady_clock::now();
                auto res = delimiterBackend_->process(
                    delimiter::StereoPlanarView{chunkInputLeft_.data(), chunkInputRight_.data(),
                                                chunkFrames_},
                    chunkOutputLeft_, chunkOutputRight_);
                auto end = std::chrono::steady_clock::now();
                double elapsed =
                    std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
                realtimeFactor = (chunkSec > 0.0f) ? (elapsed / chunkSec) : 0.0;

                inferenceOk = (res.status == delimiter::InferenceStatus::Ok);
                if (!inferenceOk) {
                    LOG_EVERY_N(WARN, 10,
                                "[AudioPipeline] Delimiter inference failed (status={}, msg='{}'), "
                                "falling back to bypass for this chunk",
                                static_cast<int>(res.status), res.message);
                }

                if (delimiterSafety_) {
                    (void)delimiterSafety_->observeInferenceResult(res);
                }
            }

            double queueSeconds = 0.0;
            if (workerInputRate_ > 0) {
                queueSeconds = static_cast<double>(inputInterleaved_.availableToRead()) /
                               (2.0 * static_cast<double>(workerInputRate_));
            }
            bool overloadTriggered = false;
            if (delimiterSafety_) {
                overloadTriggered = delimiterSafety_->observeOverload(realtimeFactor, queueSeconds);
                if (!overloadTriggered && (inferenceOk || !inferenceAttempted)) {
                    delimiterSafety_->observeHealthy();
                }
                auto status = delimiterSafety_->status();
                updateDelimiterStatus(status);
                bypassChunk = bypassChunk || (status.mode == delimiter::ProcessingMode::Bypass) ||
                              status.bypassLocked;
            }

            if (!inferenceOk || overloadTriggered) {
                bypassChunk = true;
            }

            if (bypassChunk) {
                chunkOutputLeft_ = chunkInputLeft_;
                chunkOutputRight_ = chunkInputRight_;
            }

            segmentLeft_.assign(hopFrames_, 0.0f);
            segmentRight_.assign(hopFrames_, 0.0f);
            if (!hasPrevChunk_) {
                std::copy(chunkOutputLeft_.begin(),
                          chunkOutputLeft_.begin() + static_cast<std::ptrdiff_t>(hopFrames_),
                          segmentLeft_.begin());
                std::copy(chunkOutputRight_.begin(),
                          chunkOutputRight_.begin() + static_cast<std::ptrdiff_t>(hopFrames_),
                          segmentRight_.begin());
            } else {
                for (size_t i = 0; i < overlapFrames_; ++i) {
                    segmentLeft_[i] =
                        prevOutputTailWeightedLeft_[i] + chunkOutputLeft_[i] * fadeIn_[i];
                    segmentRight_[i] =
                        prevOutputTailWeightedRight_[i] + chunkOutputRight_[i] * fadeIn_[i];
                }
                for (size_t i = overlapFrames_; i < hopFrames_; ++i) {
                    segmentLeft_[i] = chunkOutputLeft_[i];
                    segmentRight_[i] = chunkOutputRight_[i];
                }
            }

            for (size_t i = 0; i < overlapFrames_; ++i) {
                float fadeOut = fadeIn_[overlapFrames_ - 1 - i];
                prevOutputTailWeightedLeft_[i] = chunkOutputLeft_[hopFrames_ + i] * fadeOut;
                prevOutputTailWeightedRight_[i] = chunkOutputRight_[hopFrames_ + i] * fadeOut;
            }

            hasPrevChunk_ = true;

            size_t offset = 0;
            while (offset < segmentLeft_.size()) {
                size_t frames = std::min(maxFramesPerCall, segmentLeft_.size() - offset);

                downstreamInterleaved_.assign(frames * 2, 0.0f);
                for (size_t i = 0; i < frames; ++i) {
                    downstreamInterleaved_[i * 2] = segmentLeft_[offset + i];
                    downstreamInterleaved_[i * 2 + 1] = segmentRight_[offset + i];
                }

                if (deps_.buffer.playbackBuffer && deps_.running) {
                    deps_.buffer.playbackBuffer->throttleProducerIfFull(*deps_.running,
                                                                        deps_.currentOutputRate);
                }

                (void)processDirect(downstreamInterleaved_.data(), static_cast<uint32_t>(frames));
                offset += frames;

                if (workerStop_.load(std::memory_order_acquire)) {
                    break;
                }
                if (deps_.running && !deps_.running->load(std::memory_order_acquire)) {
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[AudioPipeline] High-latency worker crashed: {}", e.what());
        workerFailed_.store(true, std::memory_order_release);
    } catch (...) {
        LOG_ERROR("[AudioPipeline] High-latency worker crashed (unknown exception)");
        workerFailed_.store(true, std::memory_order_release);
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
