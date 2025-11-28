#include "gpu/partition_plan.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace ConvolutionEngine {
namespace {

int nextPow2(int value) {
    if (value <= 0) {
        return 1;
    }
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

PartitionDescriptor makeDescriptor(int taps, bool realtime, int fftMultipleOverride) {
    PartitionDescriptor desc;
    desc.taps = taps;
    desc.realtime = realtime;
    // Require enough FFT bins to accommodate taps + overlap
    const int fftMultiple = realtime ? 2 : std::max(2, fftMultipleOverride);
    const int fftTarget = std::max(taps * fftMultiple, taps + 1);
    desc.fftSize = nextPow2(fftTarget);
    desc.validOutput = std::max(1, desc.fftSize - taps + 1);
    return desc;
}

}  // namespace

PartitionPlan buildPartitionPlan(int totalTaps, int upsampleRatio,
                                 const AppConfig::PartitionedConvolutionConfig& config) {
    PartitionPlan plan;
    plan.totalTaps = totalTaps;
    if (!config.enabled || totalTaps <= 0 || upsampleRatio <= 0) {
        return plan;
    }

    plan.enabled = true;

    const int minPartitionTaps = std::max(1024, config.minPartitionTaps);
    const int maxPartitions = std::max(1, config.maxPartitions);

    int remaining = totalTaps;
    int fastLowerBound = std::min(minPartitionTaps, totalTaps);
    int fastUpperBound = std::max(fastLowerBound, totalTaps);
    int fastTaps = std::clamp(std::max(config.fastPartitionTaps, 1), fastLowerBound, fastUpperBound);

    plan.partitions.push_back(makeDescriptor(fastTaps, true, config.tailFftMultiple));
    plan.realtimeTaps = fastTaps;
    remaining -= fastTaps;
    if (remaining <= 0) {
        plan.realtimeTaps = plan.partitions.front().taps;
        return plan;
    }

    int previousTaps = fastTaps;
    int partitionsUsed = 1;

    while (remaining > 0 && partitionsUsed < maxPartitions) {
        int suggested = previousTaps * 2;
        int taps = std::min(std::max(suggested, minPartitionTaps), remaining);

        // Last allowed partition takes the rest
        if (partitionsUsed == maxPartitions - 1) {
            taps = remaining;
        }

        plan.partitions.push_back(makeDescriptor(taps, false, config.tailFftMultiple));
        remaining -= taps;
        previousTaps = taps;
        ++partitionsUsed;
    }

    if (remaining > 0) {
        plan.partitions.push_back(makeDescriptor(remaining, false, config.tailFftMultiple));
    }

    // Recompute realtime taps in case fast partition consumed entire filter
    plan.realtimeTaps = plan.partitions.front().taps;
    return plan;
}

std::string PartitionPlan::describe(int outputSampleRate) const {
    if (!enabled || partitions.empty()) {
        return "disabled";
    }

    std::ostringstream oss;
    oss << partitions.size() << " partition(s): ";

    for (size_t i = 0; i < partitions.size(); ++i) {
        const auto& part = partitions[i];
        if (i > 0) {
            oss << " | ";
        }
        oss << (part.realtime ? "fast" : "tail") << "#" << i << "=" << part.taps << " taps, FFT "
            << part.fftSize << ", valid " << part.validOutput;
        if (outputSampleRate > 0) {
            double latencyMs =
                (static_cast<double>(part.validOutput) / static_cast<double>(outputSampleRate)) *
                1000.0;
            oss << " (" << std::round(latencyMs * 10.0) / 10.0 << " ms)";
        }
    }

    return oss.str();
}

}  // namespace ConvolutionEngine


