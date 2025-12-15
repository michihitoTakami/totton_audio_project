#ifndef GPU_CONVOLUTION_KERNELS_H
#define GPU_CONVOLUTION_KERNELS_H

#include "gpu/precision_traits.h"

#include <cuda_runtime.h>
#include <cufft.h>

namespace ConvolutionEngine {

// CUDA kernel for zero-padding (insert zeros between samples for upsampling)
__global__ void zeroPadKernel(const DeviceSample* input, DeviceSample* output, int inputLength,
                              int upsampleRatio);

// CUDA kernel for complex multiplication (frequency domain)
__global__ void complexMultiplyKernel(DeviceFftComplex* data, const DeviceFftComplex* filter,
                                      int size);

// CUDA kernel for scaling after IFFT
__global__ void scaleKernel(DeviceSample* data, int size, DeviceScale scale);

// CUDA kernel to upconvert float samples to active precision samples (float->double when enabled)
__global__ void upconvertFromFloatKernel(const float* input, DeviceSample* output, int size);

// CUDA kernel for cepstrum causality window with normalization
// Applies: 1/N normalization (cuFFT doesn't normalize IFFT)
// Plus causality: c[0] unchanged, c[1..N/2-1] *= 2, c[N/2] unchanged, c[N/2+1..N-1] = 0
__global__ void applyCausalityWindowKernel(cufftDoubleReal* cepstrum, int fullN);

// CUDA kernel to exponentiate complex values
__global__ void exponentiateComplexKernel(cufftDoubleComplex* data, int n);

// CUDA kernel to convert double complex to active precision complex
__global__ void doubleToFloatComplexKernel(DeviceFftComplex* out, const cufftDoubleComplex* in,
                                           int n);

// CUDA kernel to downconvert active precision samples to float with clipping
__global__ void downconvertToFloatKernel(const DeviceSample* input, float* output, int size,
                                         float clipMin, float clipMax);

}  // namespace ConvolutionEngine

#endif  // GPU_CONVOLUTION_KERNELS_H
