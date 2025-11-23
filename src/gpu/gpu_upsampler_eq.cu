#include "convolution_engine.h"
#include "gpu/convolution_kernels.h"
#include "gpu/cuda_utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ConvolutionEngine {

// GPUUpsampler implementation - EQ Support methods

void GPUUpsampler::restoreOriginalFilter() {
    if (!d_filterFFT_A_ || !d_filterFFT_B_ || !d_originalFilterFFT_ || filterFftSize_ == 0) {
        return;
    }

    try {
        // Ping-pong: write to back buffer, then swap
        cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_)
                                    ? d_filterFFT_B_ : d_filterFFT_A_;

        Utils::checkCudaError(
            cudaMemcpy(backBuffer, d_originalFilterFFT_,
                      filterFftSize_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
            "cudaMemcpy restore original filter to back buffer"
        );

        // Synchronize to ensure copy is complete before swapping
        Utils::checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize before swap");

        // Atomic swap: now convolution kernel will use the restored filter
        d_activeFilterFFT_ = backBuffer;

        eqApplied_ = false;
        std::cout << "EQ: Restored original filter (ping-pong)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EQ: Failed to restore: " << e.what() << std::endl;
    }
}

// Apply EQ magnitude with minimum phase reconstruction.
// IMPORTANT: This function assumes the original filter is MINIMUM PHASE.
// If the original filter were linear phase, the group delay characteristics would change
// because we reconstruct minimum phase from combined magnitude |H_filter| × |H_eq|.
// This is intentional for this project (see CLAUDE.md - minimum phase is mandatory).
bool GPUUpsampler::applyEqMagnitude(const std::vector<double>& eqMagnitude) {
    if (!d_filterFFT_A_ || !d_filterFFT_B_ || !d_originalFilterFFT_ || filterFftSize_ == 0) {
        std::cerr << "EQ: Filter not initialized" << std::endl;
        return false;
    }

    if (eqMagnitude.size() != filterFftSize_) {
        std::cerr << "EQ: Magnitude size mismatch: expected " << filterFftSize_
                  << ", got " << eqMagnitude.size() << std::endl;
        return false;
    }

    // Verify persistent EQ resources are initialized
    if (!eqPlanD2Z_ || !eqPlanZ2D_ || !d_eqLogMag_ || !d_eqComplexSpec_) {
        std::cerr << "EQ: Persistent EQ resources not initialized" << std::endl;
        return false;
    }

    // Auto-normalization: prevent clipping by normalizing if max boost > 0dB
    std::vector<double> normalizedMagnitude = eqMagnitude;
    double maxMag = *std::max_element(eqMagnitude.begin(), eqMagnitude.end());
    double normalizationFactor = 1.0;

    if (maxMag > 1.0) {
        normalizationFactor = 1.0 / maxMag;
        for (size_t i = 0; i < normalizedMagnitude.size(); ++i) {
            normalizedMagnitude[i] *= normalizationFactor;
        }
        double normDb = 20.0 * std::log10(normalizationFactor);
        std::cout << "EQ: Auto-normalization applied: " << normDb << " dB "
                  << "(max boost was +" << 20.0 * std::log10(maxMag) << " dB)" << std::endl;
    }

    try {
        size_t fullN = fftSize_;
        size_t halfN = filterFftSize_;  // N/2 + 1

        // Step 1: Use host-cached original filter FFT (no D→H copy needed)

        // Step 2: Compute combined log magnitude = log(|H_original| * |H_eq|)
        std::vector<double> logMag(fullN);
        for (size_t i = 0; i < halfN; ++i) {
            double origMag = std::sqrt(h_originalFilterFft_[i].x * h_originalFilterFft_[i].x +
                                       h_originalFilterFft_[i].y * h_originalFilterFft_[i].y);
            double combined = origMag * normalizedMagnitude[i];
            if (combined < 1e-30) combined = 1e-30;  // Avoid log(0)
            logMag[i] = std::log(combined);
        }
        // Hermitian symmetry for negative frequencies
        for (size_t i = 1; i < fullN / 2; ++i) {
            logMag[fullN - i] = logMag[i];
        }

        // Step 3: Convert log magnitude to complex spectrum
        std::vector<cufftDoubleComplex> logMagComplex(halfN);
        for (size_t i = 0; i < halfN; ++i) {
            logMagComplex[i].x = logMag[i];
            logMagComplex[i].y = 0.0;
        }
        Utils::checkCudaError(
            cudaMemcpy(d_eqComplexSpec_, logMagComplex.data(),
                      halfN * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice),
            "cudaMemcpy logMag to device"
        );

        // Step 4: IFFT to get cepstrum
        if (cufftExecZ2D(eqPlanZ2D_, d_eqComplexSpec_, d_eqLogMag_) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftExecZ2D failed");
        }

        // Step 5: Apply causality window on GPU (includes 1/N normalization)
        int blockSize = 256;
        int numBlocks = (fullN + blockSize - 1) / blockSize;
        applyCausalityWindowKernel<<<numBlocks, blockSize>>>(d_eqLogMag_, fullN);

        // Step 6: FFT back
        if (cufftExecD2Z(eqPlanD2Z_, d_eqLogMag_, d_eqComplexSpec_) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftExecD2Z failed");
        }

        // Step 7: Exponentiate on GPU
        numBlocks = (halfN + blockSize - 1) / blockSize;
        exponentiateComplexKernel<<<numBlocks, blockSize>>>(d_eqComplexSpec_, halfN);

        // Step 8: Convert to float and upload to back buffer (ping-pong)
        cufftComplex* backBuffer = (d_activeFilterFFT_ == d_filterFFT_A_)
                                    ? d_filterFFT_B_ : d_filterFFT_A_;

        doubleToFloatComplexKernel<<<numBlocks, blockSize>>>(backBuffer, d_eqComplexSpec_, halfN);

        // Single sync point
        Utils::checkCudaError(cudaDeviceSynchronize(), "EQ cudaDeviceSynchronize");

        // Atomic swap
        d_activeFilterFFT_ = backBuffer;

        eqApplied_ = true;
        std::cout << "EQ: Applied with minimum phase reconstruction (GPU, ping-pong)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "EQ: Failed to apply minimum phase: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace ConvolutionEngine
