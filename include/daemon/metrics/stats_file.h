#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace daemon_metrics {

class StatsFile {
   public:
    explicit StatsFile(std::string path);

    StatsFile(const StatsFile&) = delete;
    StatsFile& operator=(const StatsFile&) = delete;

    StatsFile(StatsFile&& other) noexcept;
    StatsFile& operator=(StatsFile&& other) noexcept;

    const std::string& path() const;

    void removeIfExists() const;
    bool writeJsonAtomically(const nlohmann::json& payload) const;

   private:
    std::string path_;
};

}  // namespace daemon_metrics
