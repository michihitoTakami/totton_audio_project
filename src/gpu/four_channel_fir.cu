#include "convolution_engine.h"

#include "gpu/convolution_kernels.h"
#include "gpu/cuda_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

namespace ConvolutionEngine {

using json = nlohmann::json;
using Precision = ActivePrecisionTraits;
using Sample = DeviceSample;
using Complex = DeviceFftComplex;

namespace {
int rateFamilyToIndex(RateFamily family) {
    switch (family) {
    case RateFamily::RATE_44K:
        return 0;
    case RateFamily::RATE_48K:
        return 1;
    default:
        return -1;
    }
}

RateFamily indexToRateFamily(int idx) {
    return (idx == 0) ? RateFamily::RATE_44K : RateFamily::RATE_48K;
}

int headSizeToIndex(HeadSize size) {
    switch (size) {
    case HeadSize::XS:
        return 0;
    case HeadSize::S:
        return 1;
    case HeadSize::M:
        return 2;
    case HeadSize::L:
        return 3;
    case HeadSize::XL:
        return 4;
    default:
        return 2;
    }
}

}  // namespace

FourChannelFIR::FourChannelFIR() {
    for (auto& cfg : d_filterFFTs_) {
        cfg.fill(nullptr);
    }
    d_activeFilterFFT_.fill(nullptr);
    configLoaded_.fill(false);
}

FourChannelFIR::~FourChannelFIR() {
    cleanup();
}

int FourChannelFIR::getFamilyIndex(RateFamily family) const {
    return rateFamilyToIndex(family);
}

int FourChannelFIR::getConfigIndex(HeadSize size, RateFamily family) const {
    int familyIdx = getFamilyIndex(family);
    if (familyIdx < 0) {
        return -1;
    }
    return headSizeToIndex(size) * kRateFamilyCount + familyIdx;
}

bool FourChannelFIR::loadHrtfConfig(const std::string& binPath, const std::string& jsonPath,
                                    HeadSize size, RateFamily family, int expectedTaps) {
    std::ifstream metaFile(jsonPath);
    if (!metaFile) {
        return false;
    }
    json meta;
    try {
        metaFile >> meta;
    } catch (const std::exception& e) {
        std::cerr << "[FourChannelFIR] Failed to parse metadata " << jsonPath << ": " << e.what()
                  << std::endl;
        return false;
    }

    int taps = meta.value("n_taps", 0);
    int channels = meta.value("n_channels", 0);
    if (channels != kChannelCount || taps <= 0) {
        std::cerr << "[FourChannelFIR] Invalid metadata in " << jsonPath << " (channels=" << channels
                  << ", taps=" << taps << ")\n";
        return false;
    }

    int enforcedTaps = (expectedTaps > 0) ? expectedTaps : taps;
    if (taps != enforcedTaps) {
        std::cerr << "[FourChannelFIR] Tap mismatch: " << binPath << " expected " << enforcedTaps
                  << " got " << taps << std::endl;
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
        std::cerr << "[FourChannelFIR] File size mismatch: " << binPath << " expected "
                  << expectedBytes << " got " << fileSize << std::endl;
        return false;
    }

    std::vector<float> raw(static_cast<size_t>(kChannelCount) * filterTaps_);
    binFile.read(reinterpret_cast<char*>(raw.data()), expectedBytes);
    if (!binFile) {
        std::cerr << "[FourChannelFIR] Failed to read " << binPath << std::endl;
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

bool FourChannelFIR::setupGpuResources() {
    if (filterTaps_ <= 0 || blockSize_ <= 0) {
        return false;
    }

    try {
        stream_ = Utils::createPrioritizedStream("four_channel_fir stream", cudaStreamNonBlocking);

        size_t fftSizeNeeded =
            static_cast<size_t>(blockSize_) + static_cast<size_t>(filterTaps_) - 1;
        fftSize_ = 1;
        while (static_cast<size_t>(fftSize_) < fftSizeNeeded) {
            fftSize_ <<= 1;
        }
        overlapSize_ = filterTaps_ - 1;
        validOutputPerBlock_ = fftSize_ - overlapSize_;
        streamValidInputPerBlock_ = static_cast<size_t>(validOutputPerBlock_);

        int fftComplexSize = fftSize_ / 2 + 1;
        Utils::checkCufftError(
            cufftPlan1d(&fftPlanForward_, fftSize_, Precision::kFftTypeForward, 1),
            "cufftPlan1d forward four_channel_fir");
        Utils::checkCufftError(
            cufftPlan1d(&fftPlanInverse_, fftSize_, Precision::kFftTypeInverse, 1),
            "cufftPlan1d inverse four_channel_fir");

        Utils::checkCudaError(cudaMalloc(&d_paddedInputL_, fftSize_ * sizeof(Sample)),
                              "cudaMalloc padded input L");
        Utils::checkCudaError(cudaMalloc(&d_paddedInputR_, fftSize_ * sizeof(Sample)),
                              "cudaMalloc padded input R");
#if defined(GPU_UPSAMPLER_USE_FLOAT64)
        Utils::checkCudaError(
            cudaMalloc(&d_inputScratchL_, streamValidInputPerBlock_ * sizeof(float)),
            "cudaMalloc input scratch L");
        Utils::checkCudaError(
            cudaMalloc(&d_inputScratchR_, streamValidInputPerBlock_ * sizeof(float)),
            "cudaMalloc input scratch R");
#endif
        Utils::checkCudaError(cudaMalloc(&d_fftInputL_, fftComplexSize * sizeof(Complex)),
                              "cudaMalloc fft input L");
        Utils::checkCudaError(cudaMalloc(&d_fftInputR_, fftComplexSize * sizeof(Complex)),
                              "cudaMalloc fft input R");
        Utils::checkCudaError(cudaMalloc(&d_convLL_, fftComplexSize * sizeof(Complex)),
                              "cudaMalloc conv LL");
        Utils::checkCudaError(cudaMalloc(&d_convLR_, fftComplexSize * sizeof(Complex)),
                              "cudaMalloc conv LR");
        Utils::checkCudaError(cudaMalloc(&d_convRL_, fftComplexSize * sizeof(Complex)),
                              "cudaMalloc conv RL");
        Utils::checkCudaError(cudaMalloc(&d_convRR_, fftComplexSize * sizeof(Complex)),
                              "cudaMalloc conv RR");
        Utils::checkCudaError(cudaMalloc(&d_outputL_, fftSize_ * sizeof(Sample)),
                              "cudaMalloc output L");
        Utils::checkCudaError(cudaMalloc(&d_outputR_, fftSize_ * sizeof(Sample)),
                              "cudaMalloc output R");
        Utils::checkCudaError(cudaMalloc(&d_tempTime_, fftSize_ * sizeof(Sample)),
                              "cudaMalloc temp time");
        Utils::checkCudaError(cudaMalloc(&d_overlapInputL_, overlapSize_ * sizeof(Sample)),
                              "cudaMalloc overlap L");
        Utils::checkCudaError(cudaMalloc(&d_overlapInputR_, overlapSize_ * sizeof(Sample)),
                              "cudaMalloc overlap R");
        Utils::checkCudaError(cudaMemset(d_overlapInputL_, 0, overlapSize_ * sizeof(Sample)),
                              "cudaMemset overlap L");
        Utils::checkCudaError(cudaMemset(d_overlapInputR_, 0, overlapSize_ * sizeof(Sample)),
                              "cudaMemset overlap R");

        Sample* d_filterPadded = nullptr;
        Utils::checkCudaError(cudaMalloc(&d_filterPadded, fftSize_ * sizeof(Sample)),
                              "cudaMalloc filter padded four_channel_fir");
        for (int cfg = 0; cfg < kConfigCount; ++cfg) {
            if (!configLoaded_[cfg]) {
                continue;
            }
            for (int c = 0; c < kChannelCount; ++c) {
                Utils::checkCudaError(
                    cudaMalloc(&d_filterFFTs_[cfg][c], fftComplexSize * sizeof(Complex)),
                    "cudaMalloc filter FFT four_channel_fir");

                Utils::checkCudaError(
                    cudaMemset(d_filterPadded, 0, fftSize_ * sizeof(Sample)),
                    "cudaMemset filter padded");

                Utils::checkCudaError(copyHostToDeviceSamples<Precision>(
                                          d_filterPadded, h_filterCoeffs_[cfg][c].data(),
                                          h_filterCoeffs_[cfg][c].size()),
                                      "copy filter coeffs to device");

                Utils::checkCufftError(
                    Precision::execForward(fftPlanForward_, d_filterPadded, d_filterFFTs_[cfg][c]),
                    "cufftExec filter FFT");
            }
        }
        cudaFree(d_filterPadded);

        // Clear host coefficients after GPU upload to reduce RAM pressure.
        for (auto& cfg : h_filterCoeffs_) {
            for (auto& ch : cfg) {
                ch.clear();
                ch.shrink_to_fit();
            }
        }

        // Set active filter for initial size/family
        int activeIdx = getConfigIndex(currentHeadSize_, currentRateFamily_);
        if (activeIdx < 0 || !configLoaded_[activeIdx]) {
            // pick first available config as fallback
            for (int cfg = 0; cfg < kConfigCount; ++cfg) {
                if (configLoaded_[cfg]) {
                    activeIdx = cfg;
                    currentHeadSize_ = static_cast<HeadSize>(cfg / kRateFamilyCount);
                    currentRateFamily_ = indexToRateFamily(cfg % kRateFamilyCount);
                    break;
                }
            }
        }

        if (activeIdx < 0) {
            std::cerr << "[FourChannelFIR] No valid filter configuration loaded\n";
            return false;
        }

        for (int c = 0; c < kChannelCount; ++c) {
            d_activeFilterFFT_[c] = d_filterFFTs_[activeIdx][c];
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FourChannelFIR] setupGpuResources failed: " << e.what() << std::endl;
        return false;
    }
}

namespace {
cudaError_t copyHostFloatToDeviceSamplesAsync(Sample* dst, const float* src, size_t count,
                                             cudaStream_t stream, float* dScratch) {
#if defined(GPU_UPSAMPLER_USE_FLOAT64)
    cudaError_t err =
        cudaMemcpyAsync(dScratch, src, count * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }
    const int threadsPerBlock = 256;
    const int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
    upconvertFromFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(dScratch, dst,
                                                                     static_cast<int>(count));
    return cudaGetLastError();
#else
    (void)dScratch;
    return cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyHostToDevice, stream);
#endif
}
}  // namespace

bool FourChannelFIR::initialize(const std::string& hrtfDir, int blockSize, HeadSize initialSize,
                                RateFamily initialFamily) {
    cleanup();

    hrtfDir_ = hrtfDir;
    blockSize_ = blockSize;
    currentHeadSize_ = initialSize;
    currentRateFamily_ = initialFamily;
    enabled_ = true;

    int enforcedTaps = -1;
    bool anyLoaded = false;
    for (int sizeIdx = 0; sizeIdx < kHeadSizeCount; ++sizeIdx) {
        HeadSize hs = static_cast<HeadSize>(sizeIdx);
        for (int famIdx = 0; famIdx < kRateFamilyCount; ++famIdx) {
            RateFamily rf = indexToRateFamily(famIdx);
            std::string binPath =
                hrtfDir + "/hrtf_" + headSizeToString(hs) + "_" +
                std::string(famIdx == 0 ? "44k" : "48k") + ".bin";
            std::string jsonPath =
                hrtfDir + "/hrtf_" + headSizeToString(hs) + "_" +
                std::string(famIdx == 0 ? "44k" : "48k") + ".json";
            if (loadHrtfConfig(binPath, jsonPath, hs, rf, enforcedTaps)) {
                anyLoaded = true;
                enforcedTaps = filterTaps_;
            }
        }
    }

    if (!anyLoaded) {
        std::cerr << "[FourChannelFIR] No HRTF files found under " << hrtfDir << std::endl;
        cleanup();
        return false;
    }

    if (!setupGpuResources()) {
        cleanup();
        return false;
    }

    initialized_ = true;
    return true;
}

bool FourChannelFIR::initializeStreaming() {
    if (!initialized_ || validOutputPerBlock_ <= 0) {
        return false;
    }

    stagedOutputL_.assign(static_cast<size_t>(validOutputPerBlock_), 0.0f);
    stagedOutputR_.assign(static_cast<size_t>(validOutputPerBlock_), 0.0f);

    resetStreaming();
    streamInitialized_ = true;
    return true;
}

void FourChannelFIR::resetStreaming() {
    if (d_overlapInputL_) {
        cudaMemset(d_overlapInputL_, 0, overlapSize_ * sizeof(Sample));
    }
    if (d_overlapInputR_) {
        cudaMemset(d_overlapInputR_, 0, overlapSize_ * sizeof(Sample));
    }
}

bool FourChannelFIR::switchHeadSize(HeadSize targetSize) {
    int cfgIdx = getConfigIndex(targetSize, currentRateFamily_);
    if (cfgIdx < 0 || !configLoaded_[cfgIdx]) {
        return false;
    }
    currentHeadSize_ = targetSize;
    for (int c = 0; c < kChannelCount; ++c) {
        d_activeFilterFFT_[c] = d_filterFFTs_[cfgIdx][c];
    }
    resetStreaming();
    return true;
}

bool FourChannelFIR::switchRateFamily(RateFamily targetFamily) {
    int cfgIdx = getConfigIndex(currentHeadSize_, targetFamily);
    if (cfgIdx < 0 || !configLoaded_[cfgIdx]) {
        return false;
    }
    currentRateFamily_ = targetFamily;
    for (int c = 0; c < kChannelCount; ++c) {
        d_activeFilterFFT_[c] = d_filterFFTs_[cfgIdx][c];
    }
    resetStreaming();
    return true;
}

bool FourChannelFIR::processStreamBlock(const float* inputL, const float* inputR,
                                        size_t inputFrames, StreamFloatVector& outputL,
                                        StreamFloatVector& outputR, cudaStream_t stream,
                                        StreamFloatVector& streamInputBufferL,
                                        StreamFloatVector& streamInputBufferR,
                                        size_t& streamInputAccumulatedL,
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

    const size_t requiredL = streamInputAccumulatedL + inputFrames;
    const size_t requiredR = streamInputAccumulatedR + inputFrames;
    if (requiredL > streamInputBufferL.size() || requiredR > streamInputBufferR.size()) {
        std::cerr << "[FourChannelFIR] Input buffer too small for streaming block" << std::endl;
        outputL.clear();
        outputR.clear();
        return false;
    }

    std::copy(inputL, inputL + inputFrames,
              streamInputBufferL.begin() + static_cast<std::ptrdiff_t>(streamInputAccumulatedL));
    std::copy(inputR, inputR + inputFrames,
              streamInputBufferR.begin() + static_cast<std::ptrdiff_t>(streamInputAccumulatedR));
    streamInputAccumulatedL += inputFrames;
    streamInputAccumulatedR += inputFrames;

    if (streamInputAccumulatedL < streamValidInputPerBlock_ ||
        streamInputAccumulatedR < streamValidInputPerBlock_) {
        outputL.clear();
        outputR.clear();
        return false;
    }

    if (outputL.capacity() < static_cast<size_t>(validOutputPerBlock_) ||
        outputR.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
        std::cerr << "[FourChannelFIR] Output buffer capacity too small" << std::endl;
        outputL.clear();
        outputR.clear();
        return false;
    }

    outputL.resize(static_cast<size_t>(validOutputPerBlock_));
    outputR.resize(static_cast<size_t>(validOutputPerBlock_));

    cudaStream_t workStream = stream ? stream : stream_;
    const int fftComplexSize = fftSize_ / 2 + 1;
    const int threadsPerBlock = 256;
    const int blocksComplex = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
    const int blocksTime = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;

    // Prepare padded input with previous overlap
    Utils::checkCudaError(cudaMemsetAsync(d_paddedInputL_, 0, fftSize_ * sizeof(Sample), workStream),
                          "memset padded input L");
    Utils::checkCudaError(cudaMemsetAsync(d_paddedInputR_, 0, fftSize_ * sizeof(Sample), workStream),
                          "memset padded input R");

    if (overlapSize_ > 0) {
        Utils::checkCudaError(
            cudaMemcpyAsync(d_paddedInputL_, d_overlapInputL_, overlapSize_ * sizeof(Sample),
                            cudaMemcpyDeviceToDevice, workStream),
            "copy overlap L");
        Utils::checkCudaError(
            cudaMemcpyAsync(d_paddedInputR_, d_overlapInputR_, overlapSize_ * sizeof(Sample),
                            cudaMemcpyDeviceToDevice, workStream),
            "copy overlap R");
    }

    Utils::checkCudaError(
        copyHostFloatToDeviceSamplesAsync(d_paddedInputL_ + overlapSize_, streamInputBufferL.data(),
                                          streamValidInputPerBlock_, workStream, d_inputScratchL_),
        "copy input L to device");
    Utils::checkCudaError(
        copyHostFloatToDeviceSamplesAsync(d_paddedInputR_ + overlapSize_, streamInputBufferR.data(),
                                          streamValidInputPerBlock_, workStream, d_inputScratchR_),
        "copy input R to device");

    Utils::checkCufftError(cufftSetStream(fftPlanForward_, workStream),
                           "cufftSetStream forward four_channel_fir");
    Utils::checkCufftError(Precision::execForward(fftPlanForward_, d_paddedInputL_, d_fftInputL_),
                           "fft forward L");
    Utils::checkCufftError(Precision::execForward(fftPlanForward_, d_paddedInputR_, d_fftInputR_),
                           "fft forward R");

    // LL / LR use left input FFT
    Utils::checkCudaError(
        cudaMemcpyAsync(d_convLL_, d_fftInputL_, fftComplexSize * sizeof(Complex),
                        cudaMemcpyDeviceToDevice, workStream),
        "copy FFT LL");
    Utils::checkCudaError(
        cudaMemcpyAsync(d_convLR_, d_fftInputL_, fftComplexSize * sizeof(Complex),
                        cudaMemcpyDeviceToDevice, workStream),
        "copy FFT LR");
    complexMultiplyKernel<<<blocksComplex, threadsPerBlock, 0, workStream>>>(
        d_convLL_, d_activeFilterFFT_[0], fftComplexSize);
    complexMultiplyKernel<<<blocksComplex, threadsPerBlock, 0, workStream>>>(
        d_convLR_, d_activeFilterFFT_[1], fftComplexSize);

    // RL / RR use right input FFT
    Utils::checkCudaError(
        cudaMemcpyAsync(d_convRL_, d_fftInputR_, fftComplexSize * sizeof(Complex),
                        cudaMemcpyDeviceToDevice, workStream),
        "copy FFT RL");
    Utils::checkCudaError(
        cudaMemcpyAsync(d_convRR_, d_fftInputR_, fftComplexSize * sizeof(Complex),
                        cudaMemcpyDeviceToDevice, workStream),
        "copy FFT RR");
    complexMultiplyKernel<<<blocksComplex, threadsPerBlock, 0, workStream>>>(
        d_convRL_, d_activeFilterFFT_[2], fftComplexSize);
    complexMultiplyKernel<<<blocksComplex, threadsPerBlock, 0, workStream>>>(
        d_convRR_, d_activeFilterFFT_[3], fftComplexSize);

    Utils::checkCudaError(cudaMemsetAsync(d_outputL_, 0, fftSize_ * sizeof(Sample), workStream),
                          "memset output L");
    Utils::checkCudaError(cudaMemsetAsync(d_outputR_, 0, fftSize_ * sizeof(Sample), workStream),
                          "memset output R");

    Utils::checkCufftError(cufftSetStream(fftPlanInverse_, workStream),
                           "cufftSetStream inverse four_channel_fir");

    // LL -> outL
    Utils::checkCufftError(Precision::execInverse(fftPlanInverse_, d_convLL_, d_tempTime_),
                           "ifft LL");
    scaleKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_tempTime_, fftSize_,
                                                                Precision::scaleFactor(fftSize_));
    accumulateAddKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_outputL_, d_tempTime_,
                                                                        fftSize_);

    // RL -> outL
    Utils::checkCufftError(Precision::execInverse(fftPlanInverse_, d_convRL_, d_tempTime_),
                           "ifft RL");
    scaleKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_tempTime_, fftSize_,
                                                                Precision::scaleFactor(fftSize_));
    accumulateAddKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_outputL_, d_tempTime_,
                                                                        fftSize_);

    // LR -> outR
    Utils::checkCufftError(Precision::execInverse(fftPlanInverse_, d_convLR_, d_tempTime_),
                           "ifft LR");
    scaleKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_tempTime_, fftSize_,
                                                                Precision::scaleFactor(fftSize_));
    accumulateAddKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_outputR_, d_tempTime_,
                                                                        fftSize_);

    // RR -> outR
    Utils::checkCufftError(Precision::execInverse(fftPlanInverse_, d_convRR_, d_tempTime_),
                           "ifft RR");
    scaleKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_tempTime_, fftSize_,
                                                                Precision::scaleFactor(fftSize_));
    accumulateAddKernel<<<blocksTime, threadsPerBlock, 0, workStream>>>(d_outputR_, d_tempTime_,
                                                                        fftSize_);

    Utils::checkCudaError(cudaStreamSynchronize(workStream), "stream sync four_channel_fir");

    Utils::checkCudaError(
        copyDeviceToHostSamples<Precision>(outputL.data(), d_outputL_ + overlapSize_,
                                           validOutputPerBlock_),
        "copy output L");
    Utils::checkCudaError(
        copyDeviceToHostSamples<Precision>(outputR.data(), d_outputR_ + overlapSize_,
                                           validOutputPerBlock_),
        "copy output R");

    // Update overlap (tail of current input)
    if (overlapSize_ > 0) {
        Utils::checkCudaError(
            cudaMemcpyAsync(d_overlapInputL_, d_paddedInputL_ + streamValidInputPerBlock_,
                            overlapSize_ * sizeof(Sample), cudaMemcpyDeviceToDevice, workStream),
            "update overlap L");
        Utils::checkCudaError(
            cudaMemcpyAsync(d_overlapInputR_, d_paddedInputR_ + streamValidInputPerBlock_,
                            overlapSize_ * sizeof(Sample), cudaMemcpyDeviceToDevice, workStream),
            "update overlap R");
        Utils::checkCudaError(cudaStreamSynchronize(workStream),
                              "stream sync after overlap update four_channel_fir");
    }

    // Shift host accumulation buffers
    const size_t remainingL = streamInputAccumulatedL - streamValidInputPerBlock_;
    const size_t remainingR = streamInputAccumulatedR - streamValidInputPerBlock_;
    if (remainingL > 0) {
        std::memmove(streamInputBufferL.data(), streamInputBufferL.data() + streamValidInputPerBlock_,
                     remainingL * sizeof(float));
    }
    if (remainingR > 0) {
        std::memmove(streamInputBufferR.data(), streamInputBufferR.data() + streamValidInputPerBlock_,
                     remainingR * sizeof(float));
    }
    streamInputAccumulatedL = remainingL;
    streamInputAccumulatedR = remainingR;

    return true;
}

void FourChannelFIR::cleanup() {
    if (fftPlanForward_) {
        cufftDestroy(fftPlanForward_);
        fftPlanForward_ = 0;
    }
    if (fftPlanInverse_) {
        cufftDestroy(fftPlanInverse_);
        fftPlanInverse_ = 0;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    auto freeIf = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    freeIf(d_paddedInputL_);
    freeIf(d_paddedInputR_);
#if defined(GPU_UPSAMPLER_USE_FLOAT64)
    freeIf(d_inputScratchL_);
    freeIf(d_inputScratchR_);
#endif
    freeIf(d_fftInputL_);
    freeIf(d_fftInputR_);
    freeIf(d_convLL_);
    freeIf(d_convLR_);
    freeIf(d_convRL_);
    freeIf(d_convRR_);
    freeIf(d_outputL_);
    freeIf(d_outputR_);
    freeIf(d_tempTime_);
    freeIf(d_overlapInputL_);
    freeIf(d_overlapInputR_);

    for (auto& cfg : d_filterFFTs_) {
        for (auto& ch : cfg) {
            freeIf(ch);
        }
    }

    configLoaded_.fill(false);
    d_activeFilterFFT_.fill(nullptr);
    initialized_ = false;
    streamInitialized_ = false;
    filterTaps_ = 0;
    fftSize_ = 0;
    overlapSize_ = 0;
    validOutputPerBlock_ = 0;
    streamValidInputPerBlock_ = 0;
}

}  // namespace ConvolutionEngine
