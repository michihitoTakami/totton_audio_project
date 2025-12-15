#ifndef GPU_PRECISION_TRAITS_H
#define GPU_PRECISION_TRAITS_H

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace ConvolutionEngine {

template <typename T>
struct PrecisionTraits;

template <>
struct PrecisionTraits<float> {
    using Sample = float;
    using FftReal = float;
    using FftComplex = cufftComplex;
    using ScaleType = float;

    static constexpr cufftType kFftTypeForward = CUFFT_R2C;
    static constexpr cufftType kFftTypeInverse = CUFFT_C2R;
    static constexpr bool kIsDouble = false;

    static cufftResult execForward(cufftHandle plan, const Sample* in, FftComplex* out) {
        return cufftExecR2C(plan, const_cast<Sample*>(in), out);
    }

    static cufftResult execInverse(cufftHandle plan, FftComplex* in, Sample* out) {
        return cufftExecC2R(plan, in, out);
    }

    static constexpr ScaleType scaleFactor(std::size_t fftSize) {
        return static_cast<ScaleType>(1.0f / static_cast<float>(fftSize));
    }
};

template <>
struct PrecisionTraits<double> {
    using Sample = double;
    using FftReal = double;
    using FftComplex = cufftDoubleComplex;
    using ScaleType = double;

    static constexpr cufftType kFftTypeForward = CUFFT_D2Z;
    static constexpr cufftType kFftTypeInverse = CUFFT_Z2D;
    static constexpr bool kIsDouble = true;

    static cufftResult execForward(cufftHandle plan, const Sample* in, FftComplex* out) {
        return cufftExecD2Z(plan, const_cast<Sample*>(in), out);
    }

    static cufftResult execInverse(cufftHandle plan, FftComplex* in, Sample* out) {
        return cufftExecZ2D(plan, in, out);
    }

    static constexpr ScaleType scaleFactor(std::size_t fftSize) {
        return static_cast<ScaleType>(1.0 / static_cast<double>(fftSize));
    }
};

#ifdef GPU_UPSAMPLER_USE_FLOAT64
using ActivePrecision = double;
#else
using ActivePrecision = float;
#endif

using ActivePrecisionTraits = PrecisionTraits<ActivePrecision>;
using DeviceSample = typename ActivePrecisionTraits::Sample;
using DeviceFftReal = typename ActivePrecisionTraits::FftReal;
using DeviceFftComplex = typename ActivePrecisionTraits::FftComplex;
using DeviceScale = typename ActivePrecisionTraits::ScaleType;

template <typename Traits>
inline cudaError_t copyHostToDeviceSamples(typename Traits::Sample* dst, const float* src,
                                           std::size_t count) {
    if constexpr (Traits::kIsDouble) {
        std::vector<typename Traits::Sample> temp(count);
        for (std::size_t i = 0; i < count; ++i) {
            temp[i] = static_cast<typename Traits::Sample>(src[i]);
        }
        return cudaMemcpy(dst, temp.data(), count * sizeof(typename Traits::Sample),
                          cudaMemcpyHostToDevice);
    }
    return cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
}

template <typename Traits>
inline cudaError_t copyDeviceToHostSamples(float* dst, const typename Traits::Sample* src,
                                           std::size_t count) {
    if constexpr (Traits::kIsDouble) {
        std::vector<typename Traits::Sample> temp(count);
        auto err = cudaMemcpy(temp.data(), src, count * sizeof(typename Traits::Sample),
                              cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return err;
        }
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = static_cast<float>(temp[i]);
        }
        return cudaSuccess;
    }
    return cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost);
}

template <typename Traits>
inline std::vector<typename Traits::Sample> convertHostToPrecision(const std::vector<float>& src) {
    std::vector<typename Traits::Sample> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<typename Traits::Sample>(src[i]);
    }
    return dst;
}

// Coefficient expansion helper:
// - Keeps float32 coefficient binaries as-is
// - When GPU_UPSAMPLER_USE_FLOAT64 is enabled, expands to double on load
template <typename Traits>
inline std::vector<typename Traits::Sample> convertCoefficientsToPrecision(
    const std::vector<float>& src) {
    return convertHostToPrecision<Traits>(src);
}

}  // namespace ConvolutionEngine

#endif  // GPU_PRECISION_TRAITS_H
