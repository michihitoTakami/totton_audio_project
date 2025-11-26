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
    void process(const std::vector<float>& input, std::vector<float>& output);

   private:
    void rebuildKernel();

    float delaySamples_;
    int kernelRadius_;
    float beta_;
    std::vector<float> kernel_;
    std::vector<float> history_;
    bool kernelDirty_;
};

}  // namespace PhaseAlignment

#endif  // PHASE_ALIGNMENT_H

