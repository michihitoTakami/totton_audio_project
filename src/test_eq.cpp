/**
 * EQ Effect Verification Test
 * Processes a test signal through the GPU convolution engine with and without EQ,
 * then compares the frequency response to verify EQ is working correctly.
 */

#include "convolution_engine.h"
#include "eq_parser.h"
#include "eq_to_fir.h"

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

constexpr float PI = 3.14159265358979323846f;
constexpr int SAMPLE_RATE = 44100;
constexpr int BLOCK_SIZE = 4096;
constexpr int UPSAMPLE_RATIO = 16;

// Generate sine sweep from f_start to f_end Hz
std::vector<float> generateSweep(float duration, float f_start, float f_end) {
    int numSamples = static_cast<int>(duration * SAMPLE_RATE);
    std::vector<float> sweep(numSamples);

    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / SAMPLE_RATE;
        // Logarithmic sweep
        float phase = 2 * PI * f_start * duration / std::log(f_end / f_start) *
                      (std::pow(f_end / f_start, t / duration) - 1);
        sweep[i] = 0.5f * std::sin(phase);
    }

    // Fade in/out
    int fadeSamples = SAMPLE_RATE / 100;  // 10ms
    for (int i = 0; i < fadeSamples; ++i) {
        float fade = static_cast<float>(i) / fadeSamples;
        sweep[i] *= fade;
        sweep[numSamples - 1 - i] *= fade;
    }

    return sweep;
}

// Generate impulse
std::vector<float> generateImpulse(int numSamples = SAMPLE_RATE) {
    std::vector<float> impulse(numSamples, 0.0f);
    impulse[100] = 0.9f;  // Impulse at sample 100
    return impulse;
}

// Compute FFT magnitude spectrum (simplified DFT for analysis)
std::vector<float> computeSpectrum(const std::vector<float>& signal, int fftSize) {
    std::vector<float> spectrum(fftSize / 2);

    for (int k = 0; k < fftSize / 2; ++k) {
        float real = 0, imag = 0;
        for (int n = 0; n < std::min(static_cast<int>(signal.size()), fftSize); ++n) {
            float angle = -2 * PI * k * n / fftSize;
            real += signal[n] * std::cos(angle);
            imag += signal[n] * std::sin(angle);
        }
        spectrum[k] = 20 * std::log10(std::sqrt(real * real + imag * imag) + 1e-10f);
    }

    return spectrum;
}

// Save spectrum to CSV for plotting
void saveSpectrumCSV(const std::string& filename, const std::vector<float>& spectrum,
                     int sampleRate, int fftSize) {
    std::ofstream file(filename);
    file << "frequency,magnitude_db\n";

    for (size_t i = 1; i < spectrum.size(); ++i) {
        float freq = static_cast<float>(i) * sampleRate / fftSize;
        if (freq >= 20 && freq <= 20000) {
            file << freq << "," << spectrum[i] << "\n";
        }
    }
    file.close();
    std::cout << "Saved: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  EQ Effect Verification Test" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string filterPath = "data/coefficients/filter_44k_16x_2m_hybrid_phase.bin";
    std::string eqPath = "/home/michihito/Working/gpu_os/data/EQ/Sample_EQ.txt";

    if (argc > 1)
        eqPath = argv[1];
    if (argc > 2)
        filterPath = argv[2];

    // Initialize GPU upsampler
    std::cout << "\n1. Initializing GPU upsampler..." << std::endl;
    ConvolutionEngine::GPUUpsampler upsampler;

    if (!upsampler.initialize(filterPath, UPSAMPLE_RATIO, BLOCK_SIZE)) {
        std::cerr << "Failed to initialize GPU upsampler" << std::endl;
        return 1;
    }
    std::cout << "   GPU upsampler initialized" << std::endl;

    // Generate test signal (impulse for cleaner frequency response)
    std::cout << "\n2. Generating test impulse..." << std::endl;
    auto impulse = generateImpulse(BLOCK_SIZE * 4);
    std::cout << "   Generated " << impulse.size() << " samples" << std::endl;

    // Process WITHOUT EQ
    std::cout << "\n3. Processing WITHOUT EQ..." << std::endl;
    std::vector<float> outputNoEq;

    // Process in blocks
    for (size_t i = 0; i + BLOCK_SIZE <= impulse.size(); i += BLOCK_SIZE) {
        std::vector<float> block(impulse.begin() + i, impulse.begin() + i + BLOCK_SIZE);
        std::vector<float> output;

        if (upsampler.processChannel(block.data(), BLOCK_SIZE, output)) {
            outputNoEq.insert(outputNoEq.end(), output.begin(), output.end());
        }
    }
    std::cout << "   Output (no EQ): " << outputNoEq.size() << " samples" << std::endl;

    // Load and apply EQ
    std::cout << "\n4. Loading EQ profile: " << eqPath << std::endl;
    EQ::EqProfile eqProfile;
    if (!EQ::parseEqFile(eqPath, eqProfile)) {
        std::cerr << "Failed to parse EQ file" << std::endl;
        return 1;
    }
    std::cout << "   EQ: " << eqProfile.name << " (" << eqProfile.bands.size() << " bands, preamp "
              << eqProfile.preampDb << " dB)" << std::endl;

    // Compute EQ magnitude and apply with minimum phase reconstruction
    std::cout << "\n5. Applying EQ to filter (minimum phase)..." << std::endl;
    size_t filterFftSize = upsampler.getFilterFftSize();  // N/2+1 (R2C output)
    size_t fullFftSize = upsampler.getFullFftSize();      // N (full FFT)
    int upsampleRatio = upsampler.getUpsampleRatio();
    double outputSampleRate = SAMPLE_RATE * upsampleRatio;
    auto eqMagnitude =
        EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize, outputSampleRate, eqProfile);

    if (!upsampler.applyEqMagnitude(eqMagnitude)) {
        std::cerr << "Failed to apply EQ magnitude" << std::endl;
        return 1;
    }
    std::cout << "   EQ applied with minimum phase reconstruction" << std::endl;

    // Re-initialize for clean processing (reset overlap buffers)
    // Process WITH EQ
    std::cout << "\n6. Processing WITH EQ..." << std::endl;
    std::vector<float> outputWithEq;

    for (size_t i = 0; i + BLOCK_SIZE <= impulse.size(); i += BLOCK_SIZE) {
        std::vector<float> block(impulse.begin() + i, impulse.begin() + i + BLOCK_SIZE);
        std::vector<float> output;

        if (upsampler.processChannel(block.data(), BLOCK_SIZE, output)) {
            outputWithEq.insert(outputWithEq.end(), output.begin(), output.end());
        }
    }
    std::cout << "   Output (with EQ): " << outputWithEq.size() << " samples" << std::endl;

    // Analyze frequency response
    std::cout << "\n7. Analyzing frequency response..." << std::endl;
    int analysisFFTSize = 16384;
    // outputSampleRate already defined above as double

    auto spectrumNoEq = computeSpectrum(outputNoEq, analysisFFTSize);
    auto spectrumWithEq = computeSpectrum(outputWithEq, analysisFFTSize);

    // Save to CSV
    saveSpectrumCSV("test_output/spectrum_no_eq.csv", spectrumNoEq, outputSampleRate,
                    analysisFFTSize);
    saveSpectrumCSV("test_output/spectrum_with_eq.csv", spectrumWithEq, outputSampleRate,
                    analysisFFTSize);

    // Compute difference
    std::cout << "\n8. Computing EQ effect (difference)..." << std::endl;
    std::vector<float> difference(spectrumNoEq.size());
    for (size_t i = 0; i < spectrumNoEq.size(); ++i) {
        difference[i] = spectrumWithEq[i] - spectrumNoEq[i];
    }
    saveSpectrumCSV("test_output/spectrum_difference.csv", difference, outputSampleRate,
                    analysisFFTSize);

    // Report significant differences
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Results" << std::endl;
    std::cout << "========================================" << std::endl;

    float maxDiff = 0, minDiff = 0;
    float freqMaxDiff = 0, freqMinDiff = 0;

    for (size_t i = 1; i < difference.size(); ++i) {
        float freq = static_cast<float>(i) * outputSampleRate / analysisFFTSize;
        if (freq >= 20 && freq <= 20000) {
            if (difference[i] > maxDiff) {
                maxDiff = difference[i];
                freqMaxDiff = freq;
            }
            if (difference[i] < minDiff) {
                minDiff = difference[i];
                freqMinDiff = freq;
            }
        }
    }

    std::cout << "\nEQ Effect Summary:" << std::endl;
    std::cout << "  Max boost: " << maxDiff << " dB at " << freqMaxDiff << " Hz" << std::endl;
    std::cout << "  Max cut:   " << minDiff << " dB at " << freqMinDiff << " Hz" << std::endl;

    if (std::abs(maxDiff) < 0.5 && std::abs(minDiff) < 0.5) {
        std::cout << "\n*** WARNING: EQ effect is very small (<0.5 dB) ***" << std::endl;
        std::cout << "*** This may indicate EQ is not being applied correctly ***" << std::endl;
    } else {
        std::cout << "\n*** EQ effect confirmed! ***" << std::endl;
    }

    std::cout << "\nCSV files saved to test_output/ directory" << std::endl;
    std::cout << "Use plot_test_results.py to visualize" << std::endl;

    return 0;
}
