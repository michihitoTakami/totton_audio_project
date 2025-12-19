#pragma once

#include "daemon/core/pid_lock.h"
#include "daemon/metrics/stats_file.h"

#include <optional>
#include <string>

namespace daemon_app {

class ProcessResources {
   public:
    struct Options {
        std::string pidFilePath;
        std::string statsFilePath;
    };

    static std::optional<ProcessResources> acquire(const Options& options);

    ProcessResources(const ProcessResources&) = delete;
    ProcessResources& operator=(const ProcessResources&) = delete;

    ProcessResources(ProcessResources&&) noexcept = default;
    ProcessResources& operator=(ProcessResources&&) noexcept = default;

    ~ProcessResources();

    const daemon_core::PidLock& pidLock() const;
    const daemon_metrics::StatsFile& statsFile() const;

   private:
    ProcessResources(daemon_core::PidLock pidLock, daemon_metrics::StatsFile statsFile);

    daemon_core::PidLock pidLock_;
    daemon_metrics::StatsFile statsFile_;
};

}  // namespace daemon_app
