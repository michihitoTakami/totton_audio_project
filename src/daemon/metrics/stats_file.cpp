#include "daemon/metrics/stats_file.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace daemon_metrics {

StatsFile::StatsFile(std::string path) : path_(std::move(path)) {}

StatsFile::StatsFile(StatsFile&& other) noexcept : path_(std::move(other.path_)) {}

StatsFile& StatsFile::operator=(StatsFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    path_ = std::move(other.path_);
    return *this;
}

const std::string& StatsFile::path() const {
    return path_;
}

void StatsFile::removeIfExists() const {
    if (path_.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(path_, ec);
}

bool StatsFile::writeJsonAtomically(const nlohmann::json& payload) const {
    if (path_.empty()) {
        return false;
    }

    std::string tmpPath = path_ + ".tmp";
    std::ofstream ofs(tmpPath);
    if (!ofs) {
        return false;
    }
    ofs << payload.dump(2) << '\n';
    ofs.close();
    return (std::rename(tmpPath.c_str(), path_.c_str()) == 0);
}

}  // namespace daemon_metrics
