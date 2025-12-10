#include "gpu/convolution_kernels.h"

namespace ConvolutionEngine {

// CUDA kernel for zero-padding (insert zeros between samples for upsampling)
__global__ void zeroPadKernel(const DeviceSample* input, DeviceSample* output, int inputLength,
                              int upsampleRatio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputLength) {
        output[idx * upsampleRatio] = input[idx];
        // Zero out the intermediate samples
        for (int i = 1; i < upsampleRatio; ++i) {
            output[idx * upsampleRatio + i] = static_cast<DeviceSample>(0);
        }
    }
}

// CUDA kernel for complex multiplication (frequency domain)
__global__ void complexMultiplyKernel(DeviceFftComplex* data, const DeviceFftComplex* filter,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        auto a = data[idx].x;
        auto b = data[idx].y;
        auto c = filter[idx].x;
        auto d = filter[idx].y;

        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        data[idx].x = a * c - b * d;
        data[idx].y = a * d + b * c;
    }
}

// CUDA kernel for scaling after IFFT
__global__ void scaleKernel(DeviceSample* data, int size, DeviceScale scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

// CUDA kernel for cepstrum causality window with normalization
// Applies: 1/N normalization (cuFFT doesn't normalize IFFT)
// Plus causality: c[0] unchanged, c[1..N/2-1] *= 2, c[N/2] unchanged, c[N/2+1..N-1] = 0
__global__ void applyCausalityWindowKernel(cufftDoubleReal* cepstrum, int fullN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fullN) return;

    double invN = 1.0 / static_cast<double>(fullN);

    // Apply normalization and causality window together
    if (idx == 0 || idx == fullN / 2) {
        // DC and Nyquist: just normalize
        cepstrum[idx] *= invN;
    } else if (idx < fullN / 2) {
        // Positive time: normalize and double
        cepstrum[idx] *= 2.0 * invN;
    } else {
        // Negative time: zero out
        cepstrum[idx] = 0.0;
    }
}

// CUDA kernel to exponentiate complex values
__global__ void exponentiateComplexKernel(cufftDoubleComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double re = data[idx].x;
    double im = data[idx].y;
    double expRe = exp(re);
    data[idx].x = expRe * cos(im);
    data[idx].y = expRe * sin(im);
}

// CUDA kernel to convert double complex to float complex
__global__ void doubleToFloatComplexKernel(DeviceFftComplex* out, const cufftDoubleComplex* in,
                                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    out[idx].x = static_cast<DeviceSample>(in[idx].x);
    out[idx].y = static_cast<DeviceSample>(in[idx].y);
}

}  // namespace ConvolutionEngine
