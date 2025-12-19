#include "daemon/app/process_resources.h"

namespace daemon_app {

std::optional<ProcessResources> ProcessResources::acquire(const Options& options) {
    auto pidLock = daemon_core::PidLock::tryAcquire(options.pidFilePath);
    if (!pidLock) {
        return std::nullopt;
    }

    daemon_metrics::StatsFile statsFile(options.statsFilePath);
    statsFile.removeIfExists();

    return ProcessResources(std::move(*pidLock), std::move(statsFile));
}

ProcessResources::ProcessResources(daemon_core::PidLock pidLock,
                                   daemon_metrics::StatsFile statsFile)
    : pidLock_(std::move(pidLock)), statsFile_(std::move(statsFile)) {}

ProcessResources::~ProcessResources() {
    statsFile_.removeIfExists();
}

const daemon_core::PidLock& ProcessResources::pidLock() const {
    return pidLock_;
}

const daemon_metrics::StatsFile& ProcessResources::statsFile() const {
    return statsFile_;
}

}  // namespace daemon_app
