#include "convolution_engine.h"
#include "gpu/convolution_kernels.h"
#include "gpu/cuda_utils.h"
#include <iostream>
#include <algorithm>

namespace ConvolutionEngine {

// GPUUpsampler implementation - Streaming methods

bool GPUUpsampler::initializeStreaming() {
    if (partitionPlan_.enabled) {
        if (initializePartitionedStreaming()) {
            return true;
        }
        std::cerr << "[Partition] WARNING: Falling back to legacy streaming mode" << std::endl;
        partitionPlan_ = PartitionPlan{};
        partitionConfig_.enabled = false;
    }

    if (fftSize_ == 0 || overlapSize_ == 0) {
        std::cerr << "ERROR: GPU resources not initialized. Call initialize() first." << std::endl;
        return false;
    }

    cancelPhaseAlignedCrossfade();

    // Free existing streaming buffers if re-initializing (prevents memory leak on rate switch)
    if (streamInitialized_) {
        fprintf(stderr, "[Streaming] Re-initializing: freeing existing buffers\n");
        freeStreamingBuffers();
    }

    // Calculate valid output per block (samples at output rate that don't overlap)
    int idealValidOutput = fftSize_ - overlapSize_;
    validOutputPerBlock_ = (idealValidOutput / upsampleRatio_) * upsampleRatio_;
    streamOverlapSize_ = overlapSize_;

    fprintf(stderr, "Streaming parameters:\n");
    fprintf(stderr, "  FFT size: %d\n", fftSize_);
    fprintf(stderr, "  Filter overlap (L-1): %d\n", overlapSize_);
    fprintf(stderr, "  Ideal valid output: %d (not divisible by %d)\n", idealValidOutput, upsampleRatio_);
    fprintf(stderr, "  Actual valid output: %d (rounded to multiple of %d)\n", validOutputPerBlock_, upsampleRatio_);
    fprintf(stderr, "  Stream overlap size: %d (exact L-1)\n", streamOverlapSize_);
    fprintf(stderr, "  Zero-padding at end: %d sample(s)\n", fftSize_ - streamOverlapSize_ - validOutputPerBlock_);

    // Calculate input samples needed per block
    streamValidInputPerBlock_ = validOutputPerBlock_ / upsampleRatio_;

    // Pre-allocate GPU buffers
    size_t upsampledSize = streamValidInputPerBlock_ * upsampleRatio_;
    int fftComplexSize = fftSize_ / 2 + 1;

    Utils::checkCudaError(
        cudaMalloc(&d_streamInput_, streamValidInputPerBlock_ * sizeof(float)),
        "cudaMalloc streaming input buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamUpsampled_, upsampledSize * sizeof(float)),
        "cudaMalloc streaming upsampled buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamPadded_, fftSize_ * sizeof(float)),
        "cudaMalloc streaming padded buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamInputFFT_, fftComplexSize * sizeof(cufftComplex)),
        "cudaMalloc streaming FFT buffer"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_streamInputFFTBackup_, fftComplexSize * sizeof(cufftComplex)),
        "cudaMalloc streaming FFT backup buffer"
    );

    Utils::checkCudaError(
        cudaMalloc(&d_streamConvResult_, fftSize_ * sizeof(float)),
        "cudaMalloc streaming conv result buffer"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_streamConvResultOld_, fftSize_ * sizeof(float)),
        "cudaMalloc streaming old conv result buffer"
    );

    // Allocate device-resident overlap buffers
    Utils::checkCudaError(
        cudaMalloc(&d_overlapLeft_, streamOverlapSize_ * sizeof(float)),
        "cudaMalloc device overlap buffer (left)"
    );
    Utils::checkCudaError(
        cudaMalloc(&d_overlapRight_, streamOverlapSize_ * sizeof(float)),
        "cudaMalloc device overlap buffer (right)"
    );

    // Zero-initialize device overlap buffers
    Utils::checkCudaError(
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(float)),
        "cudaMemset device overlap buffer (left)"
    );
    Utils::checkCudaError(
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(float)),
        "cudaMemset device overlap buffer (right)"
    );

    streamInitialized_ = true;

    fprintf(stderr, "[Streaming] Initialized:\n");
    fprintf(stderr, "  - Input samples per block: %zu\n", streamValidInputPerBlock_);
    fprintf(stderr, "  - Output samples per block: %d\n", validOutputPerBlock_);
    fprintf(stderr, "  - Overlap (stream): %d samples\n", streamOverlapSize_);
    fprintf(stderr, "  - GPU streaming buffers pre-allocated\n");
    fprintf(stderr, "  - Device-resident overlap buffers allocated (no H↔D in RT path)\n");

    return true;
}

void GPUUpsampler::resetStreaming() {
    if (partitionPlan_.enabled && partitionStreamingInitialized_) {
        resetPartitionedStreaming();
        return;
    }

    // Reset device-resident overlap buffers
    if (d_overlapLeft_) {
        cudaMemset(d_overlapLeft_, 0, streamOverlapSize_ * sizeof(float));
    }
    if (d_overlapRight_) {
        cudaMemset(d_overlapRight_, 0, streamOverlapSize_ * sizeof(float));
    }
    fprintf(stderr, "[Streaming] Reset: device overlap buffers cleared\n");
}

void GPUUpsampler::freeStreamingBuffers() {
    if (!streamInitialized_) {
        return;
    }

    fprintf(stderr, "[Streaming] Freeing streaming buffers\n");

    if (d_streamInput_) {
        cudaFree(d_streamInput_);
        d_streamInput_ = nullptr;
    }
    if (d_streamUpsampled_) {
        cudaFree(d_streamUpsampled_);
        d_streamUpsampled_ = nullptr;
    }
    if (d_streamPadded_) {
        cudaFree(d_streamPadded_);
        d_streamPadded_ = nullptr;
    }
    if (d_streamInputFFT_) {
        cudaFree(d_streamInputFFT_);
        d_streamInputFFT_ = nullptr;
    }
    if (d_streamInputFFTBackup_) {
        cudaFree(d_streamInputFFTBackup_);
        d_streamInputFFTBackup_ = nullptr;
    }
    if (d_streamConvResult_) {
        cudaFree(d_streamConvResult_);
        d_streamConvResult_ = nullptr;
    }
    if (d_streamConvResultOld_) {
        cudaFree(d_streamConvResultOld_);
        d_streamConvResultOld_ = nullptr;
    }
    if (d_overlapLeft_) {
        cudaFree(d_overlapLeft_);
        d_overlapLeft_ = nullptr;
    }
    if (d_overlapRight_) {
        cudaFree(d_overlapRight_);
        d_overlapRight_ = nullptr;
    }

    streamInitialized_ = false;
    if (partitionPlan_.enabled) {
        partitionStreamingInitialized_ = false;
    }
    fprintf(stderr, "[Streaming] Streaming buffers freed\n");
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

    try {
        // Bypass mode: ratio 1 means input is already at output rate
        // Skip GPU convolution ONLY if EQ is not applied
        // When EQ is applied, we need convolution to apply the EQ filter
        if (upsampleRatio_ == 1 && !eqApplied_) {
            outputData.assign(inputData, inputData + inputFrames);
            // Clear accumulated buffer to prevent stale data when switching back to normal mode
            streamInputAccumulated = 0;
            return true;
        }

        if (!streamInitialized_) {
            std::cerr << "ERROR: Streaming mode not initialized. Call initializeStreaming() first." << std::endl;
            return false;
        }

        // 1. Accumulate input samples
        if (streamInputBuffer.empty()) {
            std::cerr << "ERROR: Streaming input buffer not allocated" << std::endl;
            return false;
        }

        size_t required = streamInputAccumulated + inputFrames;
        if (required > streamInputBuffer.size()) {
            // Upstream network sources may deliver larger bursts than the preallocated buffer.
            size_t newSize = std::max(streamInputBuffer.size() * 2, required);
            newSize = std::max(newSize, static_cast<size_t>(streamValidInputPerBlock_) * 2);
            streamInputBuffer.resize(newSize, 0.0f);
        }

        registerStreamInputBuffer(streamInputBuffer, stream);

        std::copy(inputData, inputData + inputFrames,
                  streamInputBuffer.begin() + streamInputAccumulated);
        streamInputAccumulated += inputFrames;

        // 2. Check if we have enough samples for one block
        if (streamInputAccumulated < streamValidInputPerBlock_) {
            outputData.clear();
            return false;
        }

        int adjustedOverlapSize = streamOverlapSize_;

        // 3. Process one block using pre-allocated GPU buffers
        size_t samplesToProcess = streamValidInputPerBlock_;

        // Step 3a: Transfer input to GPU
        Utils::checkCudaError(
            cudaMemcpyAsync(d_streamInput_, streamInputBuffer.data(), samplesToProcess * sizeof(float),
                           cudaMemcpyHostToDevice, stream),
            "cudaMemcpy streaming input to device"
        );

        // Step 3b: Zero-padding (upsampling)
        int threadsPerBlock = 256;
        int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
        zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            d_streamInput_, d_streamUpsampled_, samplesToProcess, upsampleRatio_
        );

        // Step 3c: Overlap-Save FFT convolution
        int fftComplexSize = fftSize_ / 2 + 1;

        // Prepare input: [overlap | new samples]
        Utils::checkCudaError(
            cudaMemsetAsync(d_streamPadded_, 0, fftSize_ * sizeof(float), stream),
            "cudaMemset streaming padded"
        );

        // Select device-resident overlap buffer
        float* d_overlap = (stream == streamLeft_) ? d_overlapLeft_ :
                           (stream == streamRight_) ? d_overlapRight_ : d_overlapLeft_;

        // Copy overlap from previous block (D2D)
        if (adjustedOverlapSize > 0) {
            Utils::checkCudaError(
                cudaMemcpyAsync(d_streamPadded_, d_overlap,
                               adjustedOverlapSize * sizeof(float), cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy streaming overlap D2D"
            );
        }

        // Copy new samples
        Utils::checkCudaError(
            cudaMemcpyAsync(d_streamPadded_ + adjustedOverlapSize, d_streamUpsampled_,
                           validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy streaming block to padded"
        );

        // FFT convolution
        Utils::checkCufftError(
            cufftSetStream(fftPlanForward_, stream),
            "cufftSetStream forward"
        );

        Utils::checkCufftError(
            cufftExecR2C(fftPlanForward_, d_streamPadded_, d_streamInputFFT_),
            "cufftExecR2C streaming"
        );

        bool crossfadeEnabled = phaseCrossfade_.active && d_streamInputFFTBackup_ &&
                                d_streamConvResultOld_ && phaseCrossfade_.previousFilter;
        if (crossfadeEnabled) {
            Utils::checkCudaError(
                cudaMemcpyAsync(d_streamInputFFTBackup_, d_streamInputFFT_,
                                fftComplexSize * sizeof(cufftComplex),
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy backup FFT for crossfade"
            );
        }

        threadsPerBlock = 256;
        blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
        complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            d_streamInputFFT_, d_activeFilterFFT_, fftComplexSize
        );

        Utils::checkCufftError(
            cufftSetStream(fftPlanInverse_, stream),
            "cufftSetStream inverse"
        );

        Utils::checkCufftError(
            cufftExecC2R(fftPlanInverse_, d_streamInputFFT_, d_streamConvResult_),
            "cufftExecC2R streaming"
        );

        // Scale
        float scale = 1.0f / fftSize_;
        int scaleBlocks = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
        scaleKernel<<<scaleBlocks, threadsPerBlock, 0, stream>>>(
            d_streamConvResult_, fftSize_, scale
        );

        // Extract valid output
        outputData.resize(validOutputPerBlock_);
        registerStreamOutputBuffer(outputData, stream);
        Utils::checkCudaError(
            cudaMemcpyAsync(outputData.data(), d_streamConvResult_ + adjustedOverlapSize,
                           validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpy streaming output to host"
        );

        if (crossfadeEnabled) {
            Utils::checkCudaError(
                cudaMemcpyAsync(d_streamInputFFT_, d_streamInputFFTBackup_,
                                fftComplexSize * sizeof(cufftComplex),
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy restore FFT for crossfade"
            );

            threadsPerBlock = 256;
            blocks = (fftComplexSize + threadsPerBlock - 1) / threadsPerBlock;
            complexMultiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                d_streamInputFFT_, phaseCrossfade_.previousFilter, fftComplexSize
            );

            Utils::checkCufftError(
                cufftExecC2R(fftPlanInverse_, d_streamInputFFT_, d_streamConvResultOld_),
                "cufftExecC2R crossfade old filter"
            );

            int scaleBlocksOld = (fftSize_ + threadsPerBlock - 1) / threadsPerBlock;
            scaleKernel<<<scaleBlocksOld, threadsPerBlock, 0, stream>>>(
                d_streamConvResultOld_, fftSize_, scale
            );

            crossfadeOldOutput_.resize(validOutputPerBlock_);
            Utils::checkCudaError(
                cudaMemcpyAsync(crossfadeOldOutput_.data(), d_streamConvResultOld_ + adjustedOverlapSize,
                                validOutputPerBlock_ * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpy crossfade old output"
            );
        }

        // Save overlap for next block using the actual upsampled input tail
        if (adjustedOverlapSize > 0) {
            if (validOutputPerBlock_ >= adjustedOverlapSize) {
                int overlapStart = validOutputPerBlock_ - adjustedOverlapSize;
                Utils::checkCudaError(
                    cudaMemcpyAsync(d_overlap, d_streamUpsampled_ + overlapStart,
                                    adjustedOverlapSize * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy streaming overlap tail"
                );
            } else {
                // Fallback: insufficient samples in this block, preserve previous overlap
                Utils::checkCudaError(
                    cudaMemcpyAsync(d_overlap, d_streamPadded_ + validOutputPerBlock_,
                                    adjustedOverlapSize * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy streaming overlap fallback"
                );
            }
        }

        // Synchronize stream
        Utils::checkCudaError(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize streaming"
        );

        if (crossfadeEnabled && phaseCrossfade_.active) {
            bool advanceProgress = false;
            if (stream == streamLeft_) {
                advanceProgress = true;
            } else if (streamLeft_ == nullptr && (stream == stream_ || stream == streamRight_)) {
                advanceProgress = true;
            }
            applyPhaseAlignedCrossfade(outputData, crossfadeOldOutput_, advanceProgress);
        }

        // 4. Shift remaining samples in input buffer
        size_t remaining = streamInputAccumulated - samplesToProcess;
        if (remaining > 0) {
            std::copy(streamInputBuffer.begin() + samplesToProcess,
                      streamInputBuffer.begin() + streamInputAccumulated,
                      streamInputBuffer.begin());
        }
        streamInputAccumulated = remaining;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in processStreamBlock: " << e.what() << std::endl;
        return false;
    }
}

bool GPUUpsampler::processPartitionedStreamBlock(
    const float* inputData, size_t inputFrames, StreamFloatVector& outputData,
    cudaStream_t stream, StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated) {
    try {
        if (upsampleRatio_ == 1 && !eqApplied_) {
            outputData.assign(inputData, inputData + inputFrames);
            streamInputAccumulated = 0;
            return true;
        }

        if (!partitionStreamingInitialized_) {
            std::cerr << "ERROR: Partitioned streaming mode not initialized. Call initializeStreaming() first."
                      << std::endl;
            return false;
        }

        if (streamInputBuffer.empty()) {
            std::cerr << "ERROR: Streaming input buffer not allocated" << std::endl;
            return false;
        }

        size_t required = streamInputAccumulated + inputFrames;
        if (required > streamInputBuffer.size()) {
            // Upstream network sources may deliver larger bursts than the preallocated buffer.
            size_t newSize = std::max(streamInputBuffer.size() * 2, required);
            newSize = std::max(newSize, static_cast<size_t>(streamValidInputPerBlock_) * 2);
            streamInputBuffer.resize(newSize, 0.0f);
        }

        registerStreamInputBuffer(streamInputBuffer, stream);

        std::copy(inputData, inputData + inputFrames,
                  streamInputBuffer.begin() + streamInputAccumulated);
        streamInputAccumulated += inputFrames;

        if (streamInputAccumulated < streamValidInputPerBlock_) {
            outputData.clear();
            return false;
        }

        size_t samplesToProcess = streamValidInputPerBlock_;

        Utils::checkCudaError(
            cudaMemcpyAsync(d_streamInput_, streamInputBuffer.data(),
                            samplesToProcess * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpy partition stream input");

        int threadsPerBlock = 256;
        int blocks = (samplesToProcess + threadsPerBlock - 1) / threadsPerBlock;
        zeroPadKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            d_streamInput_, d_streamUpsampled_, samplesToProcess, upsampleRatio_);

        int newSamples = validOutputPerBlock_;

        outputData.assign(newSamples, 0.0f);
        registerStreamOutputBuffer(outputData, stream);

        StreamFloatVector partitionTemp;
        for (auto& state : partitionStates_) {
            float* overlap =
                (stream == streamLeft_) ? state.d_overlapLeft
                                        : (stream == streamRight_) ? state.d_overlapRight
                                                                   : state.d_overlapLeft;
            if (!processPartitionBlock(state, stream, d_streamUpsampled_, newSamples, overlap,
                                       partitionTemp, outputData)) {
                return false;
            }
        }

        // Shift remaining samples in input buffer
        size_t remaining = streamInputAccumulated - samplesToProcess;
        if (remaining > 0) {
            std::copy(streamInputBuffer.begin() + samplesToProcess,
                      streamInputBuffer.begin() + streamInputAccumulated,
                      streamInputBuffer.begin());
        }
        streamInputAccumulated = remaining;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in processPartitionedStreamBlock: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace ConvolutionEngine
