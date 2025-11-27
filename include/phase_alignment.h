#ifndef PHASE_ALIGNMENT_H
#define PHASE_ALIGNMENT_H

#include <vector>

namespace PhaseAlignment {

float computeEnergyCentroid(const std::vector<float>& impulse);

class FractionalDelayLine {
   public:
    FractionalDelayLine();

    void configure(float delaySamples, int kernelRadius = 12, float beta = 8.6f);
    void reset();
    bool isBypassed() const;
    template <typename InputVector>
    void process(const InputVector& input, std::vector<float>& output);

   private:
    void rebuildKernel();

    float delaySamples_;
    int kernelRadius_;
    float beta_;
    std::vector<float> kernel_;
    std::vector<float> history_;
    bool kernelDirty_;
};

template <typename InputVector>
void FractionalDelayLine::process(const InputVector& input, std::vector<float>& output) {
    if (isBypassed()) {
        output.assign(input.begin(), input.end());
        return;
    }

    if (kernelDirty_ || kernel_.empty()) {
        rebuildKernel();
    }

    const size_t taps = kernel_.size();
    if (history_.size() != taps - 1) {
        history_.assign(taps - 1, 0.0f);
    }

    std::vector<float> extended;
    extended.reserve(history_.size() + input.size());
    extended.insert(extended.end(), history_.begin(), history_.end());
    extended.insert(extended.end(), input.begin(), input.end());

    output.resize(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        double acc = 0.0;
        const float* signalPtr = extended.data() + i;
        for (size_t k = 0; k < taps; ++k) {
            acc += static_cast<double>(kernel_[k]) * static_cast<double>(signalPtr[taps - 1 - k]);
        }
        output[i] = static_cast<float>(acc);
    }

    if (!extended.empty()) {
        history_.assign(extended.end() - static_cast<std::ptrdiff_t>(taps - 1), extended.end());
    }
}

}  // namespace PhaseAlignment

#endif  // PHASE_ALIGNMENT_H
