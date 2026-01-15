#include "convolution_engine.h"
#include "gpu/convolution_kernels.h"
#include "gpu/cuda_utils.h"
#include "logging/logger.h"
#include <algorithm>

namespace ConvolutionEngine {

using Precision = ActivePrecisionTraits;
using Sample = DeviceSample;
using Complex = DeviceFftComplex;
using ScaleType = DeviceScale;

// GPUUpsampler implementation - Streaming methods

bool GPUUpsampler::initializeStreaming() {
    if (partitionPlan_.enabled) {
        if (initializePartitionedStreaming()) {
            return true;
        }
        LOG_WARN("[Partition] Falling back to legacy streaming mode");
        partitionPlan_ = PartitionPlan{};
        partitionConfig_.enabled = false;
    }

    if (fftSize_ == 0 || overlapSize_ == 0) {
        LOG_ERROR("GPU resources not initialized. Call initialize() first.");
        return false;
    }

    cancelPhaseAlignedCrossfade();

    // Free existing streaming buffers if re-initializing (prevents memory leak on rate switch)
    if (streamInitialized_) {
        LOG_INFO("[Streaming] Re-initializing: freeing existing buffers");
        freeStreamingBuffers();
    }

    // Calculate valid output per block (samples at output rate that don't overlap)
    int idealValidOutput = fftSize_ - overlapSize_;
    validOutputPerBlock_ = (idealValidOutput / upsampleRatio_) * upsampleRatio_;
    streamOverlapSize_ = overlapSize_;

    LOG_INFO("Streaming parameters:");
    LOG_INFO("  FFT size: {}", fftSize_);
    LOG_INFO("  Filter overlap (L-1): {}", overlapSize_);
    LOG_INFO("  Ideal valid output: {} (not divisible by {})", idealValidOutput, upsampleRatio_);
    LOG_INFO("  Actual valid output: {} (rounded to multiple of {})", validOutputPerBlock_, upsampleRatio_);
    LOG_INFO("  Stream overlap size: {} (exact L-1)", streamOverlapSize_);
    LOG_INFO("  Zero-padding at end: {} sample(s)",
             fftSize_ - streamOverlapSize_ - validOutputPerBlock_);

    // Calculate input samples needed per block
    streamValidInputPerBlock_ = validOutputPerBlock_ / upsampleRatio_;

    // Pre-allocate per-channel GPU buffers (L/R split for safe async pipelining)
    size_t upsampledSize = streamValidInputPerBlock_ * upsampleRatio_;
    int fftComplexSize = fftSize_ / 2 + 1;

    for (int i = 0; i < kStreamingChannelCount; ++i) {
        auto& ch = streamingChannels_[i];
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamInput, streamValidInputPerBlock_ * sizeof(Sample)),
            "cudaMalloc streaming input buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamUpsampled, upsampledSize * sizeof(Sample)),
            "cudaMalloc streaming upsampled buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamPadded, fftSize_ * sizeof(Sample)),
            "cudaMalloc streaming padded buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamInputFFT, fftComplexSize * sizeof(Complex)),
            "cudaMalloc streaming FFT buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamInputFFTBackup, fftComplexSize * sizeof(Complex)),
            "cudaMalloc streaming FFT backup buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamConvResult, fftSize_ * sizeof(Sample)),
            "cudaMalloc streaming conv result buffer"
        );
        Utils::checkCudaError(
            cudaMalloc(&ch.d_streamConvResultOld, fftSize_ * sizeof(Sample)),
            "cudaMalloc streaming old conv result buffer"
        );

        // Host staging buffers (pinned) to avoid touching caller buffers while GPU is in-flight.
        ch.stagedInput.assign(streamValidInputPerBlock_, 0.0f);
        ch.stagedOutput.assign(validOutputPerBlock_, 0.0f);
        ch.stagedOldOutput.assign(validOutputPerBlock_, 0.0f);
        // Note:
        // - StreamFloatVector is a CudaPinnedVector<float> (already pinned).
        // - stagedOldOutput is a std::vector<float> but only used during crossfade; keep it
        //   unregistered to avoid cudaHostRegister() invalid argument on already-pinned memory.

        Utils::checkCudaError(
            cudaEventCreateWithFlags(&ch.doneEvent, cudaEventDisableTiming),
            "cudaEventCreate streaming done event"
        );
        ch.inFlight = false;
        ch.stagedOldValid = false;
    }

    // Stereo coordination event (recorded on stream_ after waiting on L/R done events)
    if (streamingStereoDoneEvent_) {
        cudaEventDestroy(streamingStereoDoneEvent_);
        streamingStereoDoneEvent_ = nullptr;
    }
    Utils::checkCudaError(
        cudaEventCreateWithFlags(&streamingStereoDoneEvent_, cudaEventDisableTiming),
        "cudaEventCreate streaming stereo done event"
    );
    streamingStereoLeftQueued_ = false;
    streamingStereoInFlight_ = false;
    streamingStereoDeliveredMask_ = 0;

    // Allocate device-resident overlap buffers
    Utils::checkCudaError(
        cudaMalloc(&d_overlapMono_, streamOverlapSize_ * sizeof(Sample)),
        "cudaMalloc device overlap buffer (mono)"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_overlapLeft_, streamOverlapSize_ * sizeof(Sample)),
        "cudaMalloc device overlap buffer (left)"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_overlapRight_, streamOverlapSize_ * sizeof(Sample)),
        "cudaMalloc device overlap buffer (right)"
    );

    // Zero-initialize device overlap buffers
    Utils::checkCudaError(
        cudaMemset(d_overlapMono_, 0, streamOverlapSize_ * sizeof(Sample)),
        "cudaMemset device overlap buffer (mono)"
    );
    Utils::checkCudaError(
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(Sample)),
        "cudaMemset device overlap buffer (left)"
    );
    Utils::checkCudaError(
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(Sample)),
        "cudaMemset device overlap buffer (right)"
    );

    streamInitialized_ = true;

    LOG_INFO("[Streaming] Initialized:");
    LOG_INFO("  - Input samples per block: {}", streamValidInputPerBlock_);
    LOG_INFO("  - Output samples per block: {}", validOutputPerBlock_);
    LOG_INFO("  - Overlap (stream): {} samples", streamOverlapSize_);
    LOG_INFO("  - GPU streaming buffers pre-allocated");
    LOG_INFO("  - Device-resident overlap buffers allocated (no H↔D in RT path)");

    return true;
}

void GPUUpsampler::resetStreaming() {
    if (partitionPlan_.enabled && partitionStreamingInitialized_) {
        resetPartitionedStreaming();
        return;
    }

    // Reset device-resident overlap buffers
    if (d_overlapMono_) {
        cudaMemset(d_overlapMono_, 0, streamOverlapSize_ * sizeof(Sample));
    }
    if (d_overlapLeft_) {
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(Sample));
    }
    if (d_overlapRight_) {
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(Sample));
    }
    for (int i = 0; i < kStreamingChannelCount; ++i) {
        streamingChannels_[i].inFlight = false;
        streamingChannels_[i].stagedOldValid = false;
    }
    streamingStereoLeftQueued_ = false;
    streamingStereoInFlight_ = false;
    streamingStereoDeliveredMask_ = 0;
    LOG_INFO("[Streaming] Reset: device overlap buffers cleared");
}

void GPUUpsampler::freeStreamingBuffers() {
    if (!streamInitialized_) {
        return;
    }

    LOG_INFO("[Streaming] Freeing streaming buffers");

    for (int i = 0; i < kStreamingChannelCount; ++i) {
        auto& ch = streamingChannels_[i];
        if (ch.d_streamInput) {
            cudaFree(ch.d_streamInput);
            ch.d_streamInput = nullptr;
        }
        if (ch.d_streamUpsampled) {
            cudaFree(ch.d_streamUpsampled);
            ch.d_streamUpsampled = nullptr;
        }
        if (ch.d_streamPadded) {
            cudaFree(ch.d_streamPadded);
            ch.d_streamPadded = nullptr;
        }
        if (ch.d_streamInputFFT) {
            cudaFree(ch.d_streamInputFFT);
            ch.d_streamInputFFT = nullptr;
        }
        if (ch.d_streamInputFFTBackup) {
            cudaFree(ch.d_streamInputFFTBackup);
            ch.d_streamInputFFTBackup = nullptr;
        }
        if (ch.d_streamConvResult) {
            cudaFree(ch.d_streamConvResult);
            ch.d_streamConvResult = nullptr;
        }
        if (ch.d_streamConvResultOld) {
            cudaFree(ch.d_streamConvResultOld);
            ch.d_streamConvResultOld = nullptr;
        }
        if (ch.doneEvent) {
            cudaEventDestroy(ch.doneEvent);
            ch.doneEvent = nullptr;
        }
        ch.stagedInput.clear();
        ch.stagedOutput.clear();
        ch.stagedOldOutput.clear();
        ch.stagedPartitionOutputs.clear();
        ch.inFlight = false;
    }

    if (d_overlapMono_) {
        cudaFree(d_overlapMono_);
        d_overlapMono_ = nullptr;
    }
    if (d_overlapLeft_) {
        cudaFree(d_overlapLeft_);
        d_overlapLeft_ = nullptr;
    }
    if (d_overlapRight_) {
        cudaFree(d_overlapRight_);
        d_overlapRight_ = nullptr;
    }

    if (streamingStereoDoneEvent_) {
        cudaEventDestroy(streamingStereoDoneEvent_);
        streamingStereoDoneEvent_ = nullptr;
    }
    streamingStereoLeftQueued_ = false;
    streamingStereoInFlight_ = false;
    streamingStereoDeliveredMask_ = 0;

    streamInitialized_ = false;
    if (partitionPlan_.enabled) {
        partitionStreamingInitialized_ = false;
    }
    LOG_INFO("[Streaming] Streaming buffers freed");
}

bool GPUUpsampler::processStreamBlock(const float* inputData,
                                       size_t inputFrames,
                                       StreamFloatVector& outputData,
                                       cudaStream_t stream,
                                       StreamFloatVector& streamInputBuffer,
                                       size_t& streamInputAccumulated) {
    if (partitionPlan_.enabled && partitionStreamingInitialized_) {
        return processPartitionedStreamBlock(inputData, inputFrames, outputData, stream,
                                             streamInputBuffer, streamInputAccumulated);
    }

    const int channelIndex = getStreamingChannelIndex(stream);
    auto& ch = streamingChannels_[channelIndex];
    const bool isLeft = (channelIndex == 1);
    const bool isRight = (channelIndex == 2);
    const bool isStereo = isLeft || isRight;
    bool producedOutput = false;
    bool producedOldValid = false;

    auto resetStreamingFlags = [&]() {
        ch.inFlight = false;
        ch.stagedOldValid = false;
        if (isStereo) {
            streamingStereoInFlight_ = false;
            streamingStereoLeftQueued_ = false;
            streamingStereoDeliveredMask_ = 0;
        }
    };

    auto outputSilence = [&]() -> bool {
        if (validOutputPerBlock_ <= 0) {
            outputData.clear();
            return false;
        }
        const size_t silenceFrames = static_cast<size_t>(validOutputPerBlock_);
        if (outputData.capacity() < silenceFrames) {
            LOG_EVERY_N(ERROR, 100,
                        "[Streaming] Output buffer capacity too small for fail-safe silence: "
                        "need {}, cap={}",
                        validOutputPerBlock_,
                        outputData.capacity());
            outputData.clear();
            return false;
        }
        outputData.resize(silenceFrames);
        std::fill(outputData.begin(), outputData.end(), 0.0f);
        return true;
    };

    auto handleRtFailure = [&](const char* context) -> bool {
        (void)context;
        resetStreamingFlags();
        streamInputAccumulated = 0;
        if (producedOutput) {
            return true;
        }
        return outputSilence();
    };

    auto checkCuda = [&](cudaError_t error, const char* context) -> bool {
        return Utils::checkCudaErrorCode(error, context) == AudioEngine::ErrorCode::OK;
    };

    auto checkCufft = [&](cufftResult result, const char* context) -> bool {
        return Utils::checkCufftErrorCode(result, context) == AudioEngine::ErrorCode::OK;
    };

        // Bypass mode: ratio 1 means input is already at output rate
        // Skip GPU convolution ONLY if EQ is not applied
        // When EQ is applied, we need convolution to apply the EQ filter
        if (upsampleRatio_ == 1 && !eqApplied_) {
            if (outputData.capacity() < inputFrames) {
                LOG_EVERY_N(ERROR, 100,
                            "[Streaming] Output buffer capacity too small (bypass): need {}, cap={}",
                            inputFrames,
                            outputData.capacity());
                outputData.clear();
                streamInputAccumulated = 0;
                return false;
            }
            outputData.resize(inputFrames);
            std::copy(inputData, inputData + inputFrames, outputData.begin());
            // Clear accumulated buffer to prevent stale data when switching back to normal mode
            streamInputAccumulated = 0;
            return true;
        }

        if (!streamInitialized_) {
            LOG_ERROR("Streaming mode not initialized. Call initializeStreaming() first.");
            return false;
        }

        // Legacy (blocking) semantics: return the current block output in the same call.
        // Non-blocking RT mode is enabled explicitly by the daemon (Issue #899).
        if (!streamingNonBlockingEnabled_) {
            // If we entered legacy mode after having in-flight work (e.g. runtime switch),
            // wait once so we don't reuse buffers concurrently.
            if (ch.inFlight && ch.doneEvent) {
                cudaEventSynchronize(ch.doneEvent);
                ch.inFlight = false;
            }

            if (streamInputBuffer.empty()) {
                LOG_EVERY_N(ERROR, 100, "Streaming input buffer not allocated");
                return false;
            }

            size_t required = streamInputAccumulated + inputFrames;
            if (required > streamInputBuffer.size()) {
                LOG_EVERY_N(ERROR, 100,
                            "[Streaming] Input buffer capacity exceeded: required={}, capacity={}",
                            required,
                            streamInputBuffer.size());
                outputData.clear();
                return false;
            }

            registerStreamInputBuffer(streamInputBuffer, stream);
            std::copy(inputData, inputData + inputFrames,
                      streamInputBuffer.begin() + streamInputAccumulated);
            streamInputAccumulated += inputFrames;

            if (streamInputAccumulated < streamValidInputPerBlock_) {
                outputData.clear();
                return false;
            }

            const int adjustedOverlapSize = streamOverlapSize_;
            const size_t samplesToProcess = streamValidInputPerBlock_;

            // Stage input so we can shift the accumulation buffer safely.
            std::copy(streamInputBuffer.begin(),
                      streamInputBuffer.begin() + static_cast<std::ptrdiff_t>(samplesToProcess),
                      ch.stagedInput.begin());

            // Step 3a: Transfer input to GPU
            if (!checkCuda(
                    copyHostToDeviceSamplesConvertedAsync(ch.d_streamInput, ch.stagedInput.data(),
                                                          samplesToProcess, stream),
                    "cudaMemcpy streaming input to device")) {
                return handleRtFailure("cudaMemcpy streaming input to device");
            }

            // Step 3b: Zero-padding (upsampling)
            int threadsPerBlock = 256;
            int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
            zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                ch.d_streamInput, ch.d_streamUpsampled, samplesToProcess, upsampleRatio_);

            // Step 3c: Overlap-Save FFT convolution
            int fftComplexSize = fftSize_ / 2 + 1;

            if (!checkCuda(
                    cudaMemsetAsync(ch.d_streamPadded, 0, fftSize_ * sizeof(Sample), stream),
                    "cudaMemset streaming padded")) {
                return handleRtFailure("cudaMemset streaming padded");
            }

            Sample* d_overlap = (channelIndex == 2) ? d_overlapRight_
                                : (channelIndex == 1) ? d_overlapLeft_
                                                      : d_overlapMono_;

            if (adjustedOverlapSize > 0) {
                if (!checkCuda(
                        cudaMemcpyAsync(ch.d_streamPadded, d_overlap,
                                        adjustedOverlapSize * sizeof(Sample),
                                        cudaMemcpyDeviceToDevice, stream),
                        "cudaMemcpy streaming overlap D2D")) {
                    return handleRtFailure("cudaMemcpy streaming overlap D2D");
                }
            }

            if (!checkCuda(
                    cudaMemcpyAsync(ch.d_streamPadded + adjustedOverlapSize, ch.d_streamUpsampled,
                                    validOutputPerBlock_ * sizeof(Sample),
                                    cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy streaming block to padded")) {
                return handleRtFailure("cudaMemcpy streaming block to padded");
            }

            if (!checkCufft(cufftSetStream(fftPlanForward_, stream), "cufftSetStream forward")) {
                return handleRtFailure("cufftSetStream forward");
            }
            if (!checkCufft(
                    Precision::execForward(fftPlanForward_, ch.d_streamPadded, ch.d_streamInputFFT),
                    "cufftExecR2C streaming")) {
                return handleRtFailure("cufftExecR2C streaming");
            }

            bool crossfadeEnabled = phaseCrossfade_.active && ch.d_streamInputFFTBackup &&
                                    ch.d_streamConvResultOld && phaseCrossfade_.previousFilter;
            if (crossfadeEnabled) {
                if (!checkCuda(
                        cudaMemcpyAsync(ch.d_streamInputFFTBackup, ch.d_streamInputFFT,
                                        fftComplexSize * sizeof(Complex),
                                        cudaMemcpyDeviceToDevice, stream),
                        "cudaMemcpy backup FFT for crossfade")) {
                    return handleRtFailure("cudaMemcpy backup FFT for crossfade");
                }
            }

            threadsPerBlock = 256;
            blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
            complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                ch.d_streamInputFFT, d_activeFilterFFT_, fftComplexSize);

            if (!checkCufft(cufftSetStream(fftPlanInverse_, stream), "cufftSetStream inverse")) {
                return handleRtFailure("cufftSetStream inverse");
            }
            if (!checkCufft(
                    Precision::execInverse(fftPlanInverse_, ch.d_streamInputFFT,
                                           ch.d_streamConvResult),
                    "cufftExecC2R streaming")) {
                return handleRtFailure("cufftExecC2R streaming");
            }

            ScaleType scale = Precision::scaleFactor(fftSize_);
            int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
            scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
                ch.d_streamConvResult, fftSize_, scale);

            if (!checkCuda(
                    downconvertToHost(ch.stagedOutput.data(),
                                      ch.d_streamConvResult + adjustedOverlapSize,
                                      validOutputPerBlock_, stream),
                    "downconvert streaming output to host")) {
                return handleRtFailure("downconvert streaming output to host");
            }

            if (crossfadeEnabled) {
                if (!checkCuda(
                        cudaMemcpyAsync(ch.d_streamInputFFT, ch.d_streamInputFFTBackup,
                                        fftComplexSize * sizeof(Complex),
                                        cudaMemcpyDeviceToDevice, stream),
                        "cudaMemcpy restore FFT for crossfade")) {
                    return handleRtFailure("cudaMemcpy restore FFT for crossfade");
                }
                complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                    ch.d_streamInputFFT, phaseCrossfade_.previousFilter, fftComplexSize);
                if (!checkCufft(
                        Precision::execInverse(fftPlanInverse_, ch.d_streamInputFFT,
                                               ch.d_streamConvResultOld),
                        "cufftExecC2R crossfade old filter")) {
                    return handleRtFailure("cufftExecC2R crossfade old filter");
                }
                int scaleBlocksOld = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
                scaleKernel<<<scaleBlocksOld, threadsPerBlock, 0, stream>>>(
                    ch.d_streamConvResultOld, fftSize_, scale);
                if (!checkCuda(
                        downconvertToHost(ch.stagedOldOutput.data(),
                                          ch.d_streamConvResultOld + adjustedOverlapSize,
                                          validOutputPerBlock_, stream),
                        "downconvert crossfade old output")) {
                    return handleRtFailure("downconvert crossfade old output");
                }
            }
            ch.stagedOldValid = crossfadeEnabled;

            if (adjustedOverlapSize > 0) {
                if (validOutputPerBlock_ >= adjustedOverlapSize) {
                    int overlapStart = validOutputPerBlock_ - adjustedOverlapSize;
                    if (!checkCuda(
                            cudaMemcpyAsync(d_overlap, ch.d_streamUpsampled + overlapStart,
                                            adjustedOverlapSize * sizeof(Sample),
                                            cudaMemcpyDeviceToDevice, stream),
                            "cudaMemcpy streaming overlap tail")) {
                        return handleRtFailure("cudaMemcpy streaming overlap tail");
                    }
                } else {
                    if (!checkCuda(
                            cudaMemcpyAsync(d_overlap, ch.d_streamPadded + validOutputPerBlock_,
                                            adjustedOverlapSize * sizeof(Sample),
                                            cudaMemcpyDeviceToDevice, stream),
                            "cudaMemcpy streaming overlap fallback")) {
                        return handleRtFailure("cudaMemcpy streaming overlap fallback");
                    }
                }
            }

            if (!checkCuda(cudaStreamSynchronize(stream),
                           "cudaStreamSynchronize legacy streaming")) {
                return handleRtFailure("cudaStreamSynchronize legacy streaming");
            }

            if (outputData.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
                LOG_EVERY_N(ERROR, 100,
                            "[Streaming] Output buffer capacity too small: need {}, cap={}",
                            validOutputPerBlock_,
                            outputData.capacity());
                outputData.clear();
                return false;
            }
            outputData.resize(validOutputPerBlock_);
            std::copy(ch.stagedOutput.begin(), ch.stagedOutput.end(), outputData.begin());

            if (ch.stagedOldValid && phaseCrossfade_.active) {
                bool advanceProgress = false;
                if (isLeft) {
                    advanceProgress = true;
                } else if (!isStereo && (stream == stream_ || stream == streamRight_)) {
                    advanceProgress = true;
                }
                applyPhaseAlignedCrossfade(outputData, ch.stagedOldOutput, advanceProgress);
                ch.stagedOldValid = false;
            }

            size_t remaining = streamInputAccumulated - samplesToProcess;
            if (remaining > 0) {
                std::copy(streamInputBuffer.begin() + samplesToProcess,
                          streamInputBuffer.begin() + streamInputAccumulated,
                          streamInputBuffer.begin());
            }
            streamInputAccumulated = remaining;
            return true;
        }

        // If a previous block has completed, deliver its staged output without blocking.
        if (isStereo) {
            if (streamingStereoInFlight_ && streamingStereoDoneEvent_ &&
                cudaEventQuery(streamingStereoDoneEvent_) == cudaSuccess) {
                const uint8_t bit = isLeft ? 0x1 : 0x2;
                if ((streamingStereoDeliveredMask_ & bit) == 0) {
                    if (outputData.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
                        LOG_EVERY_N(ERROR, 100,
                                    "[Streaming] Output buffer capacity too small: need {}, cap={}",
                                    validOutputPerBlock_,
                                    outputData.capacity());
                        outputData.clear();
                        return false;
                    }
                    outputData.resize(validOutputPerBlock_);
                    std::copy(ch.stagedOutput.begin(), ch.stagedOutput.end(), outputData.begin());
                    producedOutput = true;
                    producedOldValid = ch.stagedOldValid;
                    streamingStereoDeliveredMask_ |= bit;
                }
                if (streamingStereoDeliveredMask_ == 0x3) {
                    streamingStereoInFlight_ = false;
                    streamingStereoDeliveredMask_ = 0;
                }
            }
        } else {
            if (ch.inFlight && ch.doneEvent && cudaEventQuery(ch.doneEvent) == cudaSuccess) {
                if (outputData.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
                    LOG_EVERY_N(ERROR, 100,
                                "[Streaming] Output buffer capacity too small: need {}, cap={}",
                                validOutputPerBlock_,
                                outputData.capacity());
                    outputData.clear();
                    return false;
                }
                outputData.resize(validOutputPerBlock_);
                std::copy(ch.stagedOutput.begin(), ch.stagedOutput.end(), outputData.begin());
                producedOutput = true;
                producedOldValid = ch.stagedOldValid;
                ch.inFlight = false;
                ch.stagedOldValid = false;
            }
        }
        if (!producedOutput) {
            outputData.clear();
        }
        if (producedOutput && producedOldValid && phaseCrossfade_.active) {
            // Apply crossfade on the delivered (completed) output before we enqueue the next block
            // (which may overwrite stagedOldOutput asynchronously).
            bool advanceProgress = false;
            if (isLeft) {
                advanceProgress = true;
            } else if (!isStereo && (stream == stream_ || stream == streamRight_)) {
                advanceProgress = true;
            }
            applyPhaseAlignedCrossfade(outputData, ch.stagedOldOutput, advanceProgress);
        }

        // 1. Accumulate input samples
        if (streamInputBuffer.empty()) {
            LOG_EVERY_N(ERROR, 100, "Streaming input buffer not allocated");
            return false;
        }

        size_t required = streamInputAccumulated + inputFrames;
        if (required > streamInputBuffer.size()) {
            LOG_EVERY_N(ERROR, 100,
                        "[Streaming] Input buffer capacity exceeded: required={}, capacity={}",
                        required,
                        streamInputBuffer.size());
            outputData.clear();
            return false;
        }

        registerStreamInputBuffer(streamInputBuffer, stream);

        std::copy(inputData, inputData + inputFrames,
                  streamInputBuffer.begin() + streamInputAccumulated);
        streamInputAccumulated += inputFrames;

        // 2. Check if we have enough samples for one block
        if (streamInputAccumulated < streamValidInputPerBlock_) {
            // If we already produced output from a previous block, keep returning it while we
            // accumulate the next block.
            return producedOutput;
        }

        int adjustedOverlapSize = streamOverlapSize_;

        // 3. Process one block using pre-allocated GPU buffers.
        // For stereo, enforce lockstep: left enqueues first, right enqueues second and records a
        // joint completion event.
        size_t samplesToProcess = streamValidInputPerBlock_;
        const bool canEnqueueStereoLeft =
            isLeft && !streamingStereoInFlight_ && !streamingStereoLeftQueued_;
        const bool canEnqueueStereoRight =
            isRight && !streamingStereoInFlight_ && streamingStereoLeftQueued_;
        const bool canEnqueueMono = (!isStereo) && !ch.inFlight;
        const bool shouldEnqueue = (canEnqueueMono || canEnqueueStereoLeft || canEnqueueStereoRight);

        if (!shouldEnqueue) {
            // GPU is still in-flight for this channel pair; do not block.
            return producedOutput;
        }

        // Stage input to a pinned buffer so we can safely shift the caller accumulation buffer
        // without synchronizing the CUDA stream.
        std::copy(streamInputBuffer.begin(),
                  streamInputBuffer.begin() + static_cast<std::ptrdiff_t>(samplesToProcess),
                  ch.stagedInput.begin());

        // Step 3a: Transfer input to GPU
        if (!checkCuda(
                copyHostToDeviceSamplesConvertedAsync(ch.d_streamInput, ch.stagedInput.data(),
                                                      samplesToProcess, stream),
                "cudaMemcpy streaming input to device")) {
            return handleRtFailure("cudaMemcpy streaming input to device");
        }

        // Step 3b: Zero-padding (upsampling)
        int threadsPerBlock = 256;
        int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
        zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            ch.d_streamInput, ch.d_streamUpsampled, samplesToProcess, upsampleRatio_
        );

        // Step 3c: Overlap-Save FFT convolution
        int fftComplexSize = fftSize_ / 2 + 1;

        // Prepare input: [overlap | new samples]
        if (!checkCuda(cudaMemsetAsync(ch.d_streamPadded, 0, fftSize_ * sizeof(Sample), stream),
                       "cudaMemset streaming padded")) {
            return handleRtFailure("cudaMemset streaming padded");
        }

        // Select device-resident overlap buffer
        Sample* d_overlap = (channelIndex == 2) ? d_overlapRight_
                            : (channelIndex == 1) ? d_overlapLeft_
                                                  : d_overlapMono_;

        // Copy overlap from previous block (D2D)
        if (adjustedOverlapSize > 0) {
            if (!checkCuda(
                    cudaMemcpyAsync(ch.d_streamPadded, d_overlap,
                                   adjustedOverlapSize * sizeof(Sample),
                                   cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy streaming overlap D2D")) {
                return handleRtFailure("cudaMemcpy streaming overlap D2D");
            }
        }

        // Copy new samples
        if (!checkCuda(
                cudaMemcpyAsync(ch.d_streamPadded + adjustedOverlapSize, ch.d_streamUpsampled,
                               validOutputPerBlock_ * sizeof(Sample),
                               cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy streaming block to padded")) {
            return handleRtFailure("cudaMemcpy streaming block to padded");
        }

        // FFT convolution
        if (!checkCufft(cufftSetStream(fftPlanForward_, stream), "cufftSetStream forward")) {
            return handleRtFailure("cufftSetStream forward");
        }

        if (!checkCufft(
                Precision::execForward(fftPlanForward_, ch.d_streamPadded, ch.d_streamInputFFT),
                "cufftExecR2C streaming")) {
            return handleRtFailure("cufftExecR2C streaming");
        }

        bool crossfadeEnabled = phaseCrossfade_.active && ch.d_streamInputFFTBackup &&
                                ch.d_streamConvResultOld && phaseCrossfade_.previousFilter;
        if (crossfadeEnabled) {
            if (!checkCuda(
                    cudaMemcpyAsync(ch.d_streamInputFFTBackup, ch.d_streamInputFFT,
                                    fftComplexSize * sizeof(Complex),
                                    cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy backup FFT for crossfade")) {
                return handleRtFailure("cudaMemcpy backup FFT for crossfade");
            }
        }

        threadsPerBlock = 256;
        blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
        complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            ch.d_streamInputFFT, d_activeFilterFFT_, fftComplexSize
        );

        if (!checkCufft(cufftSetStream(fftPlanInverse_, stream), "cufftSetStream inverse")) {
            return handleRtFailure("cufftSetStream inverse");
        }

        if (!checkCufft(
                Precision::execInverse(fftPlanInverse_, ch.d_streamInputFFT, ch.d_streamConvResult),
                "cufftExecC2R streaming")) {
            return handleRtFailure("cufftExecC2R streaming");
        }

        // Scale
        ScaleType scale = Precision::scaleFactor(fftSize_);
        int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
        scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
            ch.d_streamConvResult, fftSize_, scale
        );

        // Extract valid output (staged to pinned host buffer)
        if (!checkCuda(
                downconvertToHost(ch.stagedOutput.data(),
                                  ch.d_streamConvResult + adjustedOverlapSize,
                                  validOutputPerBlock_, stream),
                "downconvert streaming output to host")) {
            return handleRtFailure("downconvert streaming output to host");
        }

        if (crossfadeEnabled) {
            if (!checkCuda(
                    cudaMemcpyAsync(ch.d_streamInputFFT, ch.d_streamInputFFTBackup,
                                    fftComplexSize * sizeof(Complex),
                                    cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy restore FFT for crossfade")) {
                return handleRtFailure("cudaMemcpy restore FFT for crossfade");
            }

            threadsPerBlock = 256;
            blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
            complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                ch.d_streamInputFFT, phaseCrossfade_.previousFilter, fftComplexSize
            );

            if (!checkCufft(
                    Precision::execInverse(fftPlanInverse_, ch.d_streamInputFFT,
                                           ch.d_streamConvResultOld),
                    "cufftExecC2R crossfade old filter")) {
                return handleRtFailure("cufftExecC2R crossfade old filter");
            }

            int scaleBlocksOld = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
            scaleKernel<<<scaleBlocksOld, threadsPerBlock, 0, stream>>>(
                ch.d_streamConvResultOld, fftSize_, scale
            );

            if (!checkCuda(
                    downconvertToHost(ch.stagedOldOutput.data(),
                                      ch.d_streamConvResultOld + adjustedOverlapSize,
                                      validOutputPerBlock_, stream),
                    "downconvert crossfade old output")) {
                return handleRtFailure("downconvert crossfade old output");
            }
        }
        ch.stagedOldValid = crossfadeEnabled;

        // Save overlap for next block using the actual upsampled input tail
        if (adjustedOverlapSize > 0) {
            if (validOutputPerBlock_ >= adjustedOverlapSize) {
                int overlapStart = validOutputPerBlock_ - adjustedOverlapSize;
                if (!checkCuda(
                        cudaMemcpyAsync(d_overlap, ch.d_streamUpsampled + overlapStart,
                                        adjustedOverlapSize * sizeof(Sample),
                                        cudaMemcpyDeviceToDevice, stream),
                        "cudaMemcpy streaming overlap tail")) {
                    return handleRtFailure("cudaMemcpy streaming overlap tail");
                }
            } else {
                // Fallback: insufficient samples in this block, preserve previous overlap
                if (!checkCuda(
                        cudaMemcpyAsync(d_overlap, ch.d_streamPadded + validOutputPerBlock_,
                                        adjustedOverlapSize * sizeof(Sample),
                                        cudaMemcpyDeviceToDevice, stream),
                        "cudaMemcpy streaming overlap fallback")) {
                    return handleRtFailure("cudaMemcpy streaming overlap fallback");
                }
            }
        }

        // Record per-channel completion event (used for stereo coordination too)
        if (!checkCuda(cudaEventRecord(ch.doneEvent, stream),
                       "cudaEventRecord streaming done")) {
            return handleRtFailure("cudaEventRecord streaming done");
        }
        if (isStereo) {
            if (isLeft) {
                streamingStereoLeftQueued_ = true;
            } else if (isRight) {
                // Build a joint completion event after both channel events are in the graph
                if (!checkCuda(
                        cudaStreamWaitEvent(stream_, streamingChannels_[1].doneEvent, 0),
                        "cudaStreamWaitEvent stereo left done")) {
                    return handleRtFailure("cudaStreamWaitEvent stereo left done");
                }
                if (!checkCuda(
                        cudaStreamWaitEvent(stream_, streamingChannels_[2].doneEvent, 0),
                        "cudaStreamWaitEvent stereo right done")) {
                    return handleRtFailure("cudaStreamWaitEvent stereo right done");
                }
                if (!checkCuda(cudaEventRecord(streamingStereoDoneEvent_, stream_),
                               "cudaEventRecord streaming stereo done")) {
                    return handleRtFailure("cudaEventRecord streaming stereo done");
                }
                streamingStereoInFlight_ = true;
                streamingStereoLeftQueued_ = false;
                streamingStereoDeliveredMask_ = 0;
            }
        } else {
            ch.inFlight = true;
        }

        // 4. Shift remaining samples in input buffer (safe: GPU reads from stagedInput)
        size_t remaining = streamInputAccumulated - samplesToProcess;
        if (remaining > 0) {
            std::copy(streamInputBuffer.begin() + samplesToProcess,
                      streamInputBuffer.begin() + streamInputAccumulated,
                      streamInputBuffer.begin());
        }
        streamInputAccumulated = remaining;

        return producedOutput;
}

bool GPUUpsampler::processPartitionedStreamBlock(
    const float* inputData, size_t inputFrames, StreamFloatVector& outputData,
    cudaStream_t stream, StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated) {
    const int channelIndex = getStreamingChannelIndex(stream);
    auto& ch = streamingChannels_[channelIndex];
    const bool isLeft = (channelIndex == 1);
    const bool isRight = (channelIndex == 2);
    const bool isStereo = isLeft || isRight;
    bool producedOutput = false;

    auto resetStreamingFlags = [&]() {
        ch.inFlight = false;
        if (isStereo) {
            streamingStereoInFlight_ = false;
            streamingStereoLeftQueued_ = false;
            streamingStereoDeliveredMask_ = 0;
        }
    };

    auto outputSilence = [&]() -> bool {
        if (validOutputPerBlock_ <= 0) {
            outputData.clear();
            return false;
        }
        const size_t silenceFrames = static_cast<size_t>(validOutputPerBlock_);
        if (outputData.capacity() < silenceFrames) {
            LOG_EVERY_N(ERROR, 100,
                        "[Partition] Output buffer capacity too small for fail-safe silence: "
                        "need {}, cap={}",
                        validOutputPerBlock_,
                        outputData.capacity());
            outputData.clear();
            return false;
        }
        outputData.resize(silenceFrames);
        std::fill(outputData.begin(), outputData.end(), 0.0f);
        return true;
    };

    auto handleRtFailure = [&](const char* context) -> bool {
        (void)context;
        resetStreamingFlags();
        streamInputAccumulated = 0;
        if (producedOutput) {
            return true;
        }
        return outputSilence();
    };

    auto checkCuda = [&](cudaError_t error, const char* context) -> bool {
        return Utils::checkCudaErrorCode(error, context) == AudioEngine::ErrorCode::OK;
    };

        if (upsampleRatio_ == 1 && !eqApplied_) {
            if (outputData.capacity() < inputFrames) {
                LOG_EVERY_N(ERROR, 100,
                            "[Partition] Output buffer capacity too small (bypass): need {}, cap={}",
                            inputFrames,
                            outputData.capacity());
                outputData.clear();
                streamInputAccumulated = 0;
                return false;
            }
            outputData.resize(inputFrames);
            std::copy(inputData, inputData + inputFrames, outputData.begin());
            streamInputAccumulated = 0;
            return true;
        }

        if (!partitionStreamingInitialized_) {
            LOG_ERROR("Partitioned streaming mode not initialized. Call initializeStreaming() first.");
            return false;
        }

        // Legacy (blocking) semantics: return the current block output in the same call.
        // This is used by offline tools/tests. RT playback enables non-blocking explicitly.
        if (!streamingNonBlockingEnabled_) {
            if (streamInputBuffer.empty()) {
                LOG_EVERY_N(ERROR, 100, "Streaming input buffer not allocated");
                return false;
            }

            size_t required = streamInputAccumulated + inputFrames;
            if (required > streamInputBuffer.size()) {
                LOG_EVERY_N(ERROR, 100,
                            "[Partition] Input buffer capacity exceeded: required={}, capacity={}",
                            required,
                            streamInputBuffer.size());
                outputData.clear();
                return false;
            }

            registerStreamInputBuffer(streamInputBuffer, stream);
            std::copy(inputData, inputData + inputFrames,
                      streamInputBuffer.begin() + streamInputAccumulated);
            streamInputAccumulated += inputFrames;

            if (streamInputAccumulated < streamValidInputPerBlock_) {
                outputData.clear();
                return false;
            }

            const size_t samplesToProcess = streamValidInputPerBlock_;
            std::copy(streamInputBuffer.begin(),
                      streamInputBuffer.begin() + static_cast<std::ptrdiff_t>(samplesToProcess),
                      ch.stagedInput.begin());

            if (!checkCuda(
                    copyHostToDeviceSamplesConvertedAsync(ch.d_streamInput, ch.stagedInput.data(),
                                                          samplesToProcess, stream),
                    "cudaMemcpy partition stream input")) {
                return handleRtFailure("cudaMemcpy partition stream input");
            }

            int threadsPerBlock = 256;
            int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
            zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                ch.d_streamInput, ch.d_streamUpsampled, samplesToProcess, upsampleRatio_);

            const int newSamples = validOutputPerBlock_;
            if (ch.stagedPartitionOutputs.size() != partitionStates_.size()) {
                LOG_ERROR("[Partition] Staged partition output size mismatch; re-init required");
                return false;
            }
            for (size_t idx = 0; idx < partitionStates_.size(); ++idx) {
                auto& state = partitionStates_[idx];
                Sample* overlap = (channelIndex == 2) ? state.d_overlapRight : state.d_overlapLeft;
                if (!processPartitionBlock(state, stream, ch.d_streamUpsampled, newSamples, overlap,
                                           ch.stagedPartitionOutputs[idx])) {
                    return handleRtFailure("processPartitionBlock");
                }
            }

            if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize partition")) {
                return handleRtFailure("cudaStreamSynchronize partition");
            }

            if (outputData.capacity() < static_cast<size_t>(newSamples)) {
                LOG_EVERY_N(ERROR, 100,
                            "[Partition] Output buffer capacity too small: need {}, cap={}",
                            newSamples,
                            outputData.capacity());
                outputData.clear();
                return handleRtFailure("partition output buffer capacity too small");
            }
            outputData.resize(static_cast<size_t>(newSamples));
            std::fill(outputData.begin(), outputData.end(), 0.0f);
            for (const auto& partOut : ch.stagedPartitionOutputs) {
                const size_t n = std::min(outputData.size(), partOut.size());
                for (size_t i = 0; i < n; ++i) {
                    outputData[i] += partOut[i];
                }
            }

            size_t remaining = streamInputAccumulated - samplesToProcess;
            if (remaining > 0) {
                std::copy(streamInputBuffer.begin() + samplesToProcess,
                          streamInputBuffer.begin() + streamInputAccumulated,
                          streamInputBuffer.begin());
            }
            streamInputAccumulated = remaining;
            return true;
        }

        // If a previous block has completed, deliver its output without blocking.
        if (isStereo) {
            if (streamingStereoInFlight_ && streamingStereoDoneEvent_ &&
                cudaEventQuery(streamingStereoDoneEvent_) == cudaSuccess) {
                const uint8_t bit = isLeft ? 0x1 : 0x2;
                if ((streamingStereoDeliveredMask_ & bit) == 0) {
                    if (outputData.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
                        LOG_EVERY_N(ERROR, 100,
                                    "[Partition] Output buffer capacity too small: need {}, cap={}",
                                    validOutputPerBlock_,
                                    outputData.capacity());
                        outputData.clear();
                        return handleRtFailure("partition output buffer capacity too small");
                    }
                    outputData.resize(static_cast<size_t>(validOutputPerBlock_));
                    std::fill(outputData.begin(), outputData.end(), 0.0f);
                    for (const auto& partOut : ch.stagedPartitionOutputs) {
                        const size_t n = std::min(outputData.size(), partOut.size());
                        for (size_t i = 0; i < n; ++i) {
                            outputData[i] += partOut[i];
                        }
                    }
                    producedOutput = true;
                    streamingStereoDeliveredMask_ |= bit;
                }
                if (streamingStereoDeliveredMask_ == 0x3) {
                    streamingStereoInFlight_ = false;
                    streamingStereoDeliveredMask_ = 0;
                }
            }
        } else {
            if (ch.inFlight && ch.doneEvent && cudaEventQuery(ch.doneEvent) == cudaSuccess) {
                if (outputData.capacity() < static_cast<size_t>(validOutputPerBlock_)) {
                    LOG_EVERY_N(ERROR, 100,
                                "[Partition] Output buffer capacity too small: need {}, cap={}",
                                validOutputPerBlock_,
                                outputData.capacity());
                    outputData.clear();
                    return handleRtFailure("partition output buffer capacity too small");
                }
                outputData.resize(static_cast<size_t>(validOutputPerBlock_));
                std::fill(outputData.begin(), outputData.end(), 0.0f);
                for (const auto& partOut : ch.stagedPartitionOutputs) {
                    const size_t n = std::min(outputData.size(), partOut.size());
                    for (size_t i = 0; i < n; ++i) {
                        outputData[i] += partOut[i];
                    }
                }
                producedOutput = true;
                ch.inFlight = false;
            }
        }
        if (!producedOutput) {
            outputData.clear();
        }

        if (streamInputBuffer.empty()) {
            LOG_EVERY_N(ERROR, 100, "Streaming input buffer not allocated");
            return false;
        }

        size_t required = streamInputAccumulated + inputFrames;
        if (required > streamInputBuffer.size()) {
            LOG_EVERY_N(ERROR, 100,
                        "[Partition] Input buffer capacity exceeded: required={}, capacity={}",
                        required,
                        streamInputBuffer.size());
            outputData.clear();
            return handleRtFailure("partition input buffer capacity exceeded");
        }

        registerStreamInputBuffer(streamInputBuffer, stream);

        std::copy(inputData, inputData + inputFrames,
                  streamInputBuffer.begin() + streamInputAccumulated);
        streamInputAccumulated += inputFrames;

        if (streamInputAccumulated < streamValidInputPerBlock_) {
            return producedOutput;
        }

        // For stereo, enforce lockstep: left enqueues first, right enqueues second and records a
        // joint completion event.
        const size_t samplesToProcess = streamValidInputPerBlock_;
        const bool canEnqueueStereoLeft =
            isLeft && !streamingStereoInFlight_ && !streamingStereoLeftQueued_;
        const bool canEnqueueStereoRight =
            isRight && !streamingStereoInFlight_ && streamingStereoLeftQueued_;
        const bool canEnqueueMono = (!isStereo) && !ch.inFlight;
        const bool shouldEnqueue = (canEnqueueMono || canEnqueueStereoLeft || canEnqueueStereoRight);

        if (!shouldEnqueue) {
            // GPU is still in-flight for this channel (or stereo pair); do not block.
            return producedOutput;
        }

        // Stage input to pinned buffer so we can safely shift the accumulation buffer.
        std::copy(streamInputBuffer.begin(),
                  streamInputBuffer.begin() + static_cast<std::ptrdiff_t>(samplesToProcess),
                  ch.stagedInput.begin());

        if (!checkCuda(
                copyHostToDeviceSamplesConvertedAsync(ch.d_streamInput, ch.stagedInput.data(),
                                                      samplesToProcess, stream),
                "cudaMemcpy partition stream input")) {
            return handleRtFailure("cudaMemcpy partition stream input");
        }

        int threadsPerBlock = 256;
        int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
        zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            ch.d_streamInput, ch.d_streamUpsampled, samplesToProcess, upsampleRatio_);

        const int newSamples = validOutputPerBlock_;
        if (static_cast<size_t>(newSamples) > ch.stagedOutput.size()) {
            LOG_ERROR("[Partition] Internal staging buffer too small: need {}, staged={}",
                      newSamples,
                      ch.stagedOutput.size());
            return false;
        }

        // Enqueue all partitions and copy each partition's output into pinned staging buffers.
        // The CPU sum happens only after doneEvent completes (next callback), avoiding RT blocking.
        if (ch.stagedPartitionOutputs.size() != partitionStates_.size()) {
            LOG_ERROR("[Partition] Staged partition output size mismatch; re-init required");
            return false;
        }
        for (size_t idx = 0; idx < partitionStates_.size(); ++idx) {
            auto& state = partitionStates_[idx];
            Sample* overlap = nullptr;
            if (channelIndex == 2) {
                overlap = state.d_overlapRight;
            } else {
                overlap = state.d_overlapLeft;  // mono/left share
            }
            if (!processPartitionBlock(state, stream, ch.d_streamUpsampled, newSamples, overlap,
                                       ch.stagedPartitionOutputs[idx])) {
                return handleRtFailure("processPartitionBlock");
            }
        }

        // Record completion event for this channel (used for stereo coordination too)
        if (!checkCuda(cudaEventRecord(ch.doneEvent, stream),
                       "cudaEventRecord partition streaming done")) {
            return handleRtFailure("cudaEventRecord partition streaming done");
        }
        if (isStereo) {
            if (isLeft) {
                streamingStereoLeftQueued_ = true;
            } else if (isRight) {
                if (!checkCuda(
                        cudaStreamWaitEvent(stream_, streamingChannels_[1].doneEvent, 0),
                        "cudaStreamWaitEvent partition stereo left done")) {
                    return handleRtFailure("cudaStreamWaitEvent partition stereo left done");
                }
                if (!checkCuda(
                        cudaStreamWaitEvent(stream_, streamingChannels_[2].doneEvent, 0),
                        "cudaStreamWaitEvent partition stereo right done")) {
                    return handleRtFailure("cudaStreamWaitEvent partition stereo right done");
                }
                if (!checkCuda(cudaEventRecord(streamingStereoDoneEvent_, stream_),
                               "cudaEventRecord partition streaming stereo done")) {
                    return handleRtFailure("cudaEventRecord partition streaming stereo done");
                }
                streamingStereoInFlight_ = true;
                streamingStereoLeftQueued_ = false;
                streamingStereoDeliveredMask_ = 0;
            }
        } else {
            ch.inFlight = true;
        }

        // Shift remaining samples in input buffer
        size_t remaining = streamInputAccumulated - samplesToProcess;
        if (remaining > 0) {
            std::copy(streamInputBuffer.begin() + samplesToProcess,
                      streamInputBuffer.begin() + streamInputAccumulated,
                      streamInputBuffer.begin());
        }
        streamInputAccumulated = remaining;

        return producedOutput;
}

}  // namespace ConvolutionEngine
