#include "gpu/backend/gpu_backend.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ConvolutionEngine::GpuBackend;

TEST(CudaBackendTest, R2CAndBackRoundTrip) {
    auto backend = createCudaBackend();
    ASSERT_NE(backend, nullptr) << "CUDA backend not available";

    const int fftSize = 32;
    const int complexCount = fftSize / 2 + 1;
    const float scale = 1.0f / static_cast<float>(fftSize);

    DeviceBuffer dInput{}, dFreq{}, dOutput{};
    ASSERT_EQ(backend->allocateDeviceBuffer(dInput, sizeof(float) * fftSize, "alloc input"),
              AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->allocateDeviceBuffer(dFreq, sizeof(float) * 2 * complexCount, "alloc freq"),
              AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->allocateDeviceBuffer(dOutput, sizeof(float) * fftSize, "alloc output"),
              AudioEngine::ErrorCode::OK);

    Stream stream{};
    ASSERT_EQ(backend->createStream(stream, "stream"), AudioEngine::ErrorCode::OK);

    FftPlan plan{};
    ASSERT_EQ(backend->createFftPlan1d(plan, fftSize, 1, FftDomain::RealToComplex, "plan"),
              AudioEngine::ErrorCode::OK);

    std::vector<float> input(fftSize);
    for (int i = 0; i < fftSize; ++i) {
        input[i] = std::sin(static_cast<float>(i));
    }
    ASSERT_EQ(backend->copy(dInput.handle.ptr, input.data(), sizeof(float) * input.size(),
                            CopyKind::HostToDevice, &stream, "h2d"),
              AudioEngine::ErrorCode::OK);

    ASSERT_EQ(backend->executeFft(plan, dInput, dFreq, FftDirection::Forward, &stream, "forward"),
              AudioEngine::ErrorCode::OK);

    ASSERT_EQ(
        backend->complexPointwiseMulScale(dFreq, dFreq, dFreq, complexCount, scale, &stream, "mul"),
        AudioEngine::ErrorCode::OK);

    ASSERT_EQ(backend->executeFft(plan, dFreq, dOutput, FftDirection::Inverse, &stream, "inverse"),
              AudioEngine::ErrorCode::OK);

    std::vector<float> output(fftSize, 0.0f);
    ASSERT_EQ(backend->copy(output.data(), dOutput.handle.ptr, sizeof(float) * output.size(),
                            CopyKind::DeviceToHost, &stream, "d2h"),
              AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->streamSynchronize(&stream, "sync"), AudioEngine::ErrorCode::OK);

    for (int i = 0; i < fftSize; ++i) {
        EXPECT_NEAR(output[i], input[i], 1e-3f) << "Mismatch at index " << i;
    }

    backend->destroyFftPlan(plan, "destroy plan");
    backend->destroyStream(stream, "destroy stream");
    backend->freeDeviceBuffer(dInput, "free input");
    backend->freeDeviceBuffer(dFreq, "free freq");
    backend->freeDeviceBuffer(dOutput, "free output");
}
