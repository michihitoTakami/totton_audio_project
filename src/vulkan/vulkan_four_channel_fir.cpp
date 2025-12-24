#include "vulkan/vulkan_four_channel_fir.h"

#include "logging/logger.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <utility>

#include "nlohmann/json.hpp"

using ConvolutionEngine::GpuBackend::CopyKind;
using ConvolutionEngine::GpuBackend::createVulkanBackend;
using ConvolutionEngine::GpuBackend::FftDirection;
using ConvolutionEngine::GpuBackend::FftDomain;
using ConvolutionEngine::GpuBackend::IGpuBackend;

namespace vulkan_backend {

namespace {
int rateFamilyToIndex(ConvolutionEngine::RateFamily family) {
    switch (family) {
    case ConvolutionEngine::RateFamily::RATE_44K:
        return 0;
    case ConvolutionEngine::RateFamily::RATE_48K:
        return 1;
    default:
        return -1;
    }
}

ConvolutionEngine::RateFamily indexToRateFamily(int idx) {
    return (idx == 0) ? ConvolutionEngine::RateFamily::RATE_44K
                      : ConvolutionEngine::RateFamily::RATE_48K;
}

size_t nextPow2(size_t v) {
    if (v == 0) {
        return 1;
    }
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}
}  // namespace

VulkanFourChannelFIR::VulkanFourChannelFIR() = default;

VulkanFourChannelFIR::~VulkanFourChannelFIR() {
    cleanup();
}

int VulkanFourChannelFIR::getFamilyIndex(ConvolutionEngine::RateFamily family) const {
    return rateFamilyToIndex(family);
}

int VulkanFourChannelFIR::getConfigIndex(ConvolutionEngine::HeadSize size,
                                         ConvolutionEngine::RateFamily family) const {
    int familyIdx = getFamilyIndex(family);
    if (familyIdx < 0) {
        return -1;
    }
    return static_cast<int>(size) * kRateFamilyCount + familyIdx;
}

bool VulkanFourChannelFIR::loadHrtfConfig(const std::string& binPath, const std::string& jsonPath,
                                          ConvolutionEngine::HeadSize size,
                                          ConvolutionEngine::RateFamily family, int expectedTaps) {
    std::ifstream metaFile(jsonPath);
    if (!metaFile) {
        return false;
    }

    nlohmann::json meta;
    try {
        metaFile >> meta;
    } catch (const std::exception& e) {
        LOG_ERROR("[VulkanFourChannelFIR] Failed to parse metadata {}: {}", jsonPath, e.what());
        return false;
    }

    int taps = meta.value("n_taps", 0);
    int channels = meta.value("n_channels", 0);
    if (channels != kChannelCount || taps <= 0) {
        LOG_ERROR("[VulkanFourChannelFIR] Invalid metadata in {} (channels={}, taps={})", jsonPath,
                  channels, taps);
        return false;
    }

    int enforcedTaps = (expectedTaps > 0) ? expectedTaps : taps;
    if (taps != enforcedTaps) {
        LOG_ERROR("[VulkanFourChannelFIR] Tap mismatch: {} expected {} got {}", binPath,
                  enforcedTaps, taps);
        return false;
    }
    filterTaps_ = enforcedTaps;

    std::ifstream binFile(binPath, std::ios::binary);
    if (!binFile) {
        return false;
    }
    binFile.seekg(0, std::ios::end);
    size_t fileSize = static_cast<size_t>(binFile.tellg());
    binFile.seekg(0, std::ios::beg);

    size_t expectedBytes = static_cast<size_t>(kChannelCount) * filterTaps_ * sizeof(float);
    if (fileSize != expectedBytes) {
        LOG_ERROR("[VulkanFourChannelFIR] File size mismatch: {} expected {} got {}", binPath,
                  expectedBytes, fileSize);
        return false;
    }

    std::vector<float> raw(static_cast<size_t>(kChannelCount) * filterTaps_);
    binFile.read(reinterpret_cast<char*>(raw.data()), expectedBytes);
    if (!binFile) {
        LOG_ERROR("[VulkanFourChannelFIR] Failed to read {}", binPath);
        return false;
    }

    bool channelMajor = meta.value("storage_format", "channel_major_v1") == "channel_major_v1";

    int cfgIdx = getConfigIndex(size, family);
    if (cfgIdx < 0) {
        return false;
    }
    for (int c = 0; c < kChannelCount; ++c) {
        h_filterCoeffs_[cfgIdx][c].assign(filterTaps_, 0.0f);
    }

    if (channelMajor) {
        for (int c = 0; c < kChannelCount; ++c) {
            size_t base = static_cast<size_t>(c) * filterTaps_;
            std::copy(raw.begin() + base, raw.begin() + base + filterTaps_,
                      h_filterCoeffs_[cfgIdx][c].begin());
        }
    } else {
        for (int tap = 0; tap < filterTaps_; ++tap) {
            size_t base = static_cast<size_t>(tap) * kChannelCount;
            for (int c = 0; c < kChannelCount; ++c) {
                h_filterCoeffs_[cfgIdx][c][tap] = raw[base + c];
            }
        }
    }

    configLoaded_[cfgIdx] = true;
    return true;
}

bool VulkanFourChannelFIR::setupBackendResources() {
    if (!backend_ || filterTaps_ <= 0 || blockSize_ <= 0) {
        return false;
    }

    size_t fftSizeNeeded = static_cast<size_t>(blockSize_) + static_cast<size_t>(filterTaps_) - 1;
    fftSize_ = static_cast<int>(nextPow2(fftSizeNeeded));
    overlapSize_ = filterTaps_ - 1;
    validOutputPerBlock_ = fftSize_ - overlapSize_;
    streamValidInputPerBlock_ = static_cast<size_t>(validOutputPerBlock_);
    complexCount_ = static_cast<size_t>(fftSize_ / 2 + 1);

    const size_t realBytes = static_cast<size_t>(fftSize_) * sizeof(float);
    const size_t complexBytes = complexCount_ * 2 * sizeof(float);

    auto check = [&](AudioEngine::ErrorCode ec, const char* context) -> bool {
        if (ec != AudioEngine::ErrorCode::OK) {
            LOG_ERROR("[VulkanFourChannelFIR] {} failed: {}", context, static_cast<int>(ec));
            return false;
        }
        return true;
    };

    if (!check(backend_->allocateDeviceBuffer(d_paddedInputL_, realBytes, "alloc padded L"),
               "alloc padded L") ||
        !check(backend_->allocateDeviceBuffer(d_paddedInputR_, realBytes, "alloc padded R"),
               "alloc padded R") ||
        !check(backend_->allocateDeviceBuffer(d_fftInputL_, complexBytes, "alloc fft L"),
               "alloc fft L") ||
        !check(backend_->allocateDeviceBuffer(d_fftInputR_, complexBytes, "alloc fft R"),
               "alloc fft R") ||
        !check(backend_->allocateDeviceBuffer(d_convLL_, complexBytes, "alloc conv LL"),
               "alloc conv LL") ||
        !check(backend_->allocateDeviceBuffer(d_convLR_, complexBytes, "alloc conv LR"),
               "alloc conv LR") ||
        !check(backend_->allocateDeviceBuffer(d_convRL_, complexBytes, "alloc conv RL"),
               "alloc conv RL") ||
        !check(backend_->allocateDeviceBuffer(d_convRR_, complexBytes, "alloc conv RR"),
               "alloc conv RR") ||
        !check(backend_->allocateDeviceBuffer(d_timeDomain_, realBytes, "alloc time domain"),
               "alloc time domain") ||
        !check(backend_->allocateDeviceBuffer(d_filterPadded_, realBytes, "alloc filter padded"),
               "alloc filter padded") ||
        !check(backend_->allocateDeviceBuffer(d_filterWork_, complexBytes, "alloc filter work"),
               "alloc filter work")) {
        return false;
    }

    if (!check(backend_->createFftPlan1d(fftPlanForward_, fftSize_, 1, FftDomain::RealToComplex,
                                         "fft forward plan"),
               "create forward plan") ||
        !check(backend_->createFftPlan1d(fftPlanInverse_, fftSize_, 1, FftDomain::ComplexToReal,
                                         "fft inverse plan"),
               "create inverse plan")) {
        return false;
    }

    // Upload filters and pre-compute spectra for every available config.
    hostPaddedL_.assign(static_cast<size_t>(fftSize_), 0.0f);

    for (int cfg = 0; cfg < kConfigCount; ++cfg) {
        if (!configLoaded_[cfg]) {
            continue;
        }
        for (int c = 0; c < kChannelCount; ++c) {
            if (h_filterCoeffs_[cfg][c].empty()) {
                continue;
            }
            std::fill(hostPaddedL_.begin(), hostPaddedL_.end(), 0.0f);
            std::copy(h_filterCoeffs_[cfg][c].begin(), h_filterCoeffs_[cfg][c].end(),
                      hostPaddedL_.begin());

            if (!check(backend_->copy(d_filterPadded_.handle.ptr, hostPaddedL_.data(), realBytes,
                                      CopyKind::HostToDevice, nullptr, "upload filter"),
                       "upload filter") ||
                !check(backend_->executeFft(fftPlanForward_, d_filterPadded_, d_filterWork_,
                                            FftDirection::Forward, nullptr, "fft filter"),
                       "fft filter")) {
                return false;
            }

            if (!check(backend_->allocateDeviceBuffer(d_filterFFTs_[cfg][c], complexBytes,
                                                      "alloc filter spectrum"),
                       "alloc filter spectrum")) {
                return false;
            }

            if (!check(backend_->copy(d_filterFFTs_[cfg][c].handle.ptr, d_filterWork_.handle.ptr,
                                      complexBytes, CopyKind::DeviceToDevice, nullptr,
                                      "copy filter spectrum"),
                       "copy filter spectrum")) {
                return false;
            }
        }
    }

    // Clear host coefficients to reduce memory footprint.
    for (auto& cfg : h_filterCoeffs_) {
        for (auto& ch : cfg) {
            ch.clear();
            ch.shrink_to_fit();
        }
    }

    // Select active filter
    int activeIdx = getConfigIndex(currentHeadSize_, currentRateFamily_);
    if (activeIdx < 0 || !configLoaded_[activeIdx]) {
        for (int cfg = 0; cfg < kConfigCount; ++cfg) {
            if (configLoaded_[cfg]) {
                activeIdx = cfg;
                currentHeadSize_ = static_cast<ConvolutionEngine::HeadSize>(cfg / kRateFamilyCount);
                currentRateFamily_ = indexToRateFamily(cfg % kRateFamilyCount);
                break;
            }
        }
    }
    if (activeIdx < 0) {
        LOG_ERROR("[VulkanFourChannelFIR] No valid filter configuration loaded");
        return false;
    }
    for (int c = 0; c < kChannelCount; ++c) {
        activeFilterFFT_[c] = &d_filterFFTs_[activeIdx][c];
    }

    // Release temporary buffers used only during setup.
    backend_->freeDeviceBuffer(d_filterPadded_, "free filter padded");
    backend_->freeDeviceBuffer(d_filterWork_, "free filter work");
    d_filterPadded_.handle.ptr = nullptr;
    d_filterPadded_.bytes = 0;
    d_filterWork_.handle.ptr = nullptr;
    d_filterWork_.bytes = 0;

    hostPaddedR_.assign(static_cast<size_t>(fftSize_), 0.0f);
    hostTimeWork_.assign(static_cast<size_t>(fftSize_), 0.0f);
    overlapInputL_.assign(static_cast<size_t>(overlapSize_), 0.0f);
    overlapInputR_.assign(static_cast<size_t>(overlapSize_), 0.0f);

    return true;
}

bool VulkanFourChannelFIR::initialize(const std::string& hrtfDir, int blockSize,
                                      ConvolutionEngine::HeadSize initialSize,
                                      ConvolutionEngine::RateFamily initialFamily) {
    cleanup();

    hrtfDir_ = hrtfDir;
    blockSize_ = blockSize;
    currentHeadSize_ = initialSize;
    currentRateFamily_ = initialFamily;
    enabled_ = true;

    backend_ = createVulkanBackend();
    if (!backend_) {
        LOG_ERROR("[VulkanFourChannelFIR] Failed to create Vulkan backend");
        return false;
    }

    int enforcedTaps = -1;
    bool anyLoaded = false;
    for (int sizeIdx = 0; sizeIdx < kHeadSizeCount; ++sizeIdx) {
        ConvolutionEngine::HeadSize hs = static_cast<ConvolutionEngine::HeadSize>(sizeIdx);
        for (int famIdx = 0; famIdx < kRateFamilyCount; ++famIdx) {
            ConvolutionEngine::RateFamily rf = indexToRateFamily(famIdx);
            std::string binPath = hrtfDir + "/hrtf_" + ConvolutionEngine::headSizeToString(hs) +
                                  "_" + std::string(famIdx == 0 ? "44k" : "48k") + ".bin";
            std::string jsonPath = hrtfDir + "/hrtf_" + ConvolutionEngine::headSizeToString(hs) +
                                   "_" + std::string(famIdx == 0 ? "44k" : "48k") + ".json";
            if (loadHrtfConfig(binPath, jsonPath, hs, rf, enforcedTaps)) {
                anyLoaded = true;
                enforcedTaps = filterTaps_;
            }
        }
    }

    if (!anyLoaded) {
        LOG_ERROR("[VulkanFourChannelFIR] No HRTF files found under {}", hrtfDir);
        cleanup();
        return false;
    }

    if (!setupBackendResources()) {
        cleanup();
        return false;
    }

    initialized_ = true;
    return true;
}

bool VulkanFourChannelFIR::initializeStreaming() {
    if (!initialized_ || validOutputPerBlock_ <= 0) {
        return false;
    }

    stagedOutputL_.assign(static_cast<size_t>(validOutputPerBlock_), 0.0f);
    stagedOutputR_.assign(static_cast<size_t>(validOutputPerBlock_), 0.0f);
    resetStreaming();
    streamInitialized_ = true;
    return true;
}

void VulkanFourChannelFIR::resetStreaming() {
    if (!overlapInputL_.empty()) {
        std::fill(overlapInputL_.begin(), overlapInputL_.end(), 0.0f);
    }
    if (!overlapInputR_.empty()) {
        std::fill(overlapInputR_.begin(), overlapInputR_.end(), 0.0f);
    }
}

bool VulkanFourChannelFIR::switchHeadSize(ConvolutionEngine::HeadSize targetSize) {
    int cfgIdx = getConfigIndex(targetSize, currentRateFamily_);
    if (cfgIdx < 0 || !configLoaded_[cfgIdx]) {
        return false;
    }
    currentHeadSize_ = targetSize;
    for (int c = 0; c < kChannelCount; ++c) {
        activeFilterFFT_[c] = &d_filterFFTs_[cfgIdx][c];
    }
    resetStreaming();
    return true;
}

bool VulkanFourChannelFIR::switchRateFamily(ConvolutionEngine::RateFamily targetFamily) {
    int cfgIdx = getConfigIndex(currentHeadSize_, targetFamily);
    if (cfgIdx < 0 || !configLoaded_[cfgIdx]) {
        return false;
    }
    currentRateFamily_ = targetFamily;
    for (int c = 0; c < kChannelCount; ++c) {
        activeFilterFFT_[c] = &d_filterFFTs_[cfgIdx][c];
    }
    resetStreaming();
    return true;
}

bool VulkanFourChannelFIR::processStreamBlock(
    const float* inputL, const float* inputR, size_t inputFrames,
    ConvolutionEngine::StreamFloatVector& outputL, ConvolutionEngine::StreamFloatVector& outputR,
    cudaStream_t /*stream*/, ConvolutionEngine::StreamFloatVector& streamInputBufferL,
    ConvolutionEngine::StreamFloatVector& streamInputBufferR, size_t& streamInputAccumulatedL,
    size_t& streamInputAccumulatedR) {
    if (!initialized_ || !streamInitialized_) {
        outputL.clear();
        outputR.clear();
        return false;
    }

    if (!inputL || !inputR || inputFrames == 0) {
        outputL.clear();
        outputR.clear();
        return false;
    }

    if (!enabled_) {
        if (outputL.capacity() < inputFrames || outputR.capacity() < inputFrames) {
            outputL.clear();
            outputR.clear();
            return false;
        }
        outputL.assign(inputL, inputL + inputFrames);
        outputR.assign(inputR, inputR + inputFrames);
        streamInputAccumulatedL = 0;
        streamInputAccumulatedR = 0;
        return true;
    }

    auto handleRtFailure = [&]() -> bool {
        outputL.clear();
        outputR.clear();
        streamInputAccumulatedL = 0;
        streamInputAccumulatedR = 0;
        return false;
    };

    const size_t requiredL = streamInputAccumulatedL + inputFrames;
    const size_t requiredR = streamInputAccumulatedR + inputFrames;
    if (requiredL > streamInputBufferL.size() || requiredR > streamInputBufferR.size()) {
        const size_t sizeL = streamInputBufferL.size();
        const size_t sizeR = streamInputBufferR.size();
        const size_t kept = std::min({inputFrames, sizeL, sizeR});
        LOG_EVERY_N(ERROR, 100,
                    "[VulkanFourChannelFIR] Input buffer too small (requiredL={}, requiredR={}, "
                    "sizeL={}, sizeR={}, inputFrames={}). Dropping accumulated stream state "
                    "(keep={} frames).",
                    requiredL, requiredR, sizeL, sizeR, inputFrames, kept);

        streamInputAccumulatedL = 0;
        streamInputAccumulatedR = 0;

        if (kept > 0 && inputL && inputR) {
            const float* tailL = inputL + (inputFrames - kept);
            const float* tailR = inputR + (inputFrames - kept);
            std::copy(tailL, tailL + kept, streamInputBufferL.begin());
            std::copy(tailR, tailR + kept, streamInputBufferR.begin());
            streamInputAccumulatedL = kept;
            streamInputAccumulatedR = kept;
        }
    } else {
        std::copy(
            inputL, inputL + inputFrames,
            streamInputBufferL.begin() + static_cast<std::ptrdiff_t>(streamInputAccumulatedL));
        std::copy(
            inputR, inputR + inputFrames,
            streamInputBufferR.begin() + static_cast<std::ptrdiff_t>(streamInputAccumulatedR));
        streamInputAccumulatedL += inputFrames;
        streamInputAccumulatedR += inputFrames;
    }

    if (streamInputAccumulatedL < streamValidInputPerBlock_ ||
        streamInputAccumulatedR < streamValidInputPerBlock_) {
        outputL.clear();
        outputR.clear();
        return false;
    }

    if (outputL.capacity() < static_cast<size_t>(validOutputPerBlock_) ||
        outputR.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
        LOG_EVERY_N(ERROR, 100, "[VulkanFourChannelFIR] Output buffer capacity too small");
        outputL.clear();
        outputR.clear();
        return false;
    }

    outputL.resize(static_cast<size_t>(validOutputPerBlock_));
    outputR.resize(static_cast<size_t>(validOutputPerBlock_));
    std::fill(stagedOutputL_.begin(), stagedOutputL_.end(), 0.0f);
    std::fill(stagedOutputR_.begin(), stagedOutputR_.end(), 0.0f);

    auto check = [&](AudioEngine::ErrorCode ec, const char* context) -> bool {
        if (ec != AudioEngine::ErrorCode::OK) {
            LOG_ERROR("[VulkanFourChannelFIR] {} failed: {}", context, static_cast<int>(ec));
            return false;
        }
        return true;
    };

    // Prepare padded input
    std::fill(hostPaddedL_.begin(), hostPaddedL_.end(), 0.0f);
    std::fill(hostPaddedR_.begin(), hostPaddedR_.end(), 0.0f);
    if (overlapSize_ > 0) {
        std::copy(overlapInputL_.begin(), overlapInputL_.end(), hostPaddedL_.begin());
        std::copy(overlapInputR_.begin(), overlapInputR_.end(), hostPaddedR_.begin());
    }
    std::copy(streamInputBufferL.begin(), streamInputBufferL.begin() + streamValidInputPerBlock_,
              hostPaddedL_.begin() + overlapSize_);
    std::copy(streamInputBufferR.begin(), streamInputBufferR.begin() + streamValidInputPerBlock_,
              hostPaddedR_.begin() + overlapSize_);

    const size_t realBytes = static_cast<size_t>(fftSize_) * sizeof(float);
    if (!check(backend_->copy(d_paddedInputL_.handle.ptr, hostPaddedL_.data(), realBytes,
                              CopyKind::HostToDevice, nullptr, "upload input L"),
               "upload input L") ||
        !check(backend_->copy(d_paddedInputR_.handle.ptr, hostPaddedR_.data(), realBytes,
                              CopyKind::HostToDevice, nullptr, "upload input R"),
               "upload input R")) {
        return handleRtFailure();
    }

    if (!check(backend_->executeFft(fftPlanForward_, d_paddedInputL_, d_fftInputL_,
                                    FftDirection::Forward, nullptr, "fft forward L"),
               "fft forward L") ||
        !check(backend_->executeFft(fftPlanForward_, d_paddedInputR_, d_fftInputR_,
                                    FftDirection::Forward, nullptr, "fft forward R"),
               "fft forward R")) {
        return handleRtFailure();
    }

    const size_t complexBytes = complexCount_ * 2 * sizeof(float);
    if (!check(backend_->copy(d_convLL_.handle.ptr, d_fftInputL_.handle.ptr, complexBytes,
                              CopyKind::DeviceToDevice, nullptr, "copy FFT LL"),
               "copy FFT LL") ||
        !check(backend_->copy(d_convLR_.handle.ptr, d_fftInputL_.handle.ptr, complexBytes,
                              CopyKind::DeviceToDevice, nullptr, "copy FFT LR"),
               "copy FFT LR") ||
        !check(backend_->copy(d_convRL_.handle.ptr, d_fftInputR_.handle.ptr, complexBytes,
                              CopyKind::DeviceToDevice, nullptr, "copy FFT RL"),
               "copy FFT RL") ||
        !check(backend_->copy(d_convRR_.handle.ptr, d_fftInputR_.handle.ptr, complexBytes,
                              CopyKind::DeviceToDevice, nullptr, "copy FFT RR"),
               "copy FFT RR")) {
        return handleRtFailure();
    }

    auto mul = [&](DeviceBuffer& out, const DeviceBuffer& a, const DeviceBuffer* filter,
                   const char* label) -> bool {
        if (!filter || !filter->handle.ptr) {
            LOG_ERROR("[VulkanFourChannelFIR] Active filter missing for {}", label);
            return false;
        }
        return check(backend_->complexPointwiseMulScale(out, a, *filter, complexCount_, 1.0f,
                                                        nullptr, label),
                     label);
    };

    if (!mul(d_convLL_, d_convLL_, activeFilterFFT_[0], "mul LL") ||
        !mul(d_convLR_, d_convLR_, activeFilterFFT_[1], "mul LR") ||
        !mul(d_convRL_, d_convRL_, activeFilterFFT_[2], "mul RL") ||
        !mul(d_convRR_, d_convRR_, activeFilterFFT_[3], "mul RR")) {
        return handleRtFailure();
    }

    auto inverseAndAccumulate = [&](DeviceBuffer& freqBuf,
                                    ConvolutionEngine::StreamFloatVector& dst,
                                    const char* label) -> bool {
        if (!check(backend_->executeFft(fftPlanInverse_, freqBuf, d_timeDomain_,
                                        FftDirection::Inverse, nullptr, label),
                   label)) {
            return false;
        }
        if (!check(backend_->copy(hostTimeWork_.data(), d_timeDomain_.handle.ptr, realBytes,
                                  CopyKind::DeviceToHost, nullptr, "copy time domain"),
                   "copy time domain")) {
            return false;
        }
        for (size_t i = 0; i < static_cast<size_t>(validOutputPerBlock_); ++i) {
            dst[i] += hostTimeWork_[static_cast<size_t>(overlapSize_) + i];
        }
        return true;
    };

    if (!inverseAndAccumulate(d_convLL_, stagedOutputL_, "ifft LL") ||
        !inverseAndAccumulate(d_convRL_, stagedOutputL_, "ifft RL") ||
        !inverseAndAccumulate(d_convLR_, stagedOutputR_, "ifft LR") ||
        !inverseAndAccumulate(d_convRR_, stagedOutputR_, "ifft RR")) {
        return handleRtFailure();
    }

    if (overlapSize_ > 0) {
        std::copy(hostPaddedL_.begin() + streamValidInputPerBlock_,
                  hostPaddedL_.begin() + streamValidInputPerBlock_ + overlapSize_,
                  overlapInputL_.begin());
        std::copy(hostPaddedR_.begin() + streamValidInputPerBlock_,
                  hostPaddedR_.begin() + streamValidInputPerBlock_ + overlapSize_,
                  overlapInputR_.begin());
    }

    std::copy(stagedOutputL_.begin(), stagedOutputL_.end(), outputL.begin());
    std::copy(stagedOutputR_.begin(), stagedOutputR_.end(), outputR.begin());

    const size_t remainingL = streamInputAccumulatedL - streamValidInputPerBlock_;
    const size_t remainingR = streamInputAccumulatedR - streamValidInputPerBlock_;
    if (remainingL > 0) {
        std::memmove(streamInputBufferL.data(),
                     streamInputBufferL.data() + streamValidInputPerBlock_,
                     remainingL * sizeof(float));
    }
    if (remainingR > 0) {
        std::memmove(streamInputBufferR.data(),
                     streamInputBufferR.data() + streamValidInputPerBlock_,
                     remainingR * sizeof(float));
    }
    streamInputAccumulatedL = remainingL;
    streamInputAccumulatedR = remainingR;

    return true;
}

size_t VulkanFourChannelFIR::getStreamValidInputPerBlock() const {
    return streamValidInputPerBlock_;
}

size_t VulkanFourChannelFIR::getValidOutputPerBlock() const {
    return static_cast<size_t>(validOutputPerBlock_);
}

int VulkanFourChannelFIR::getFilterTaps() const {
    return filterTaps_;
}

ConvolutionEngine::HeadSize VulkanFourChannelFIR::getCurrentHeadSize() const {
    return currentHeadSize_;
}

ConvolutionEngine::RateFamily VulkanFourChannelFIR::getCurrentRateFamily() const {
    return currentRateFamily_;
}

bool VulkanFourChannelFIR::isInitialized() const {
    return initialized_;
}

void VulkanFourChannelFIR::setEnabled(bool enabled) {
    enabled_ = enabled;
}

bool VulkanFourChannelFIR::isEnabled() const {
    return enabled_;
}

void VulkanFourChannelFIR::cleanup() {
    if (backend_) {
        backend_->destroyFftPlan(fftPlanForward_, "destroy forward plan");
        backend_->destroyFftPlan(fftPlanInverse_, "destroy inverse plan");

        auto freeBuf = [&](DeviceBuffer& buf, const char* ctx) {
            if (buf.handle.ptr) {
                backend_->freeDeviceBuffer(buf, ctx);
            }
            buf.handle.ptr = nullptr;
            buf.bytes = 0;
        };

        freeBuf(d_paddedInputL_, "free padded L");
        freeBuf(d_paddedInputR_, "free padded R");
        freeBuf(d_fftInputL_, "free fft L");
        freeBuf(d_fftInputR_, "free fft R");
        freeBuf(d_convLL_, "free conv LL");
        freeBuf(d_convLR_, "free conv LR");
        freeBuf(d_convRL_, "free conv RL");
        freeBuf(d_convRR_, "free conv RR");
        freeBuf(d_timeDomain_, "free time domain");
        freeBuf(d_filterPadded_, "free filter padded");
        freeBuf(d_filterWork_, "free filter work");

        for (auto& cfg : d_filterFFTs_) {
            for (auto& ch : cfg) {
                freeBuf(ch, "free filter spectrum");
            }
        }
    }

    backend_.reset();
    configLoaded_.fill(false);
    activeFilterFFT_.fill(nullptr);
    initialized_ = false;
    streamInitialized_ = false;
    filterTaps_ = 0;
    fftSize_ = 0;
    overlapSize_ = 0;
    validOutputPerBlock_ = 0;
    streamValidInputPerBlock_ = 0;
    complexCount_ = 0;
    overlapInputL_.clear();
    overlapInputR_.clear();
    hostPaddedL_.clear();
    hostPaddedR_.clear();
    hostTimeWork_.clear();
    stagedOutputL_.clear();
    stagedOutputR_.clear();
}

}  // namespace vulkan_backend
