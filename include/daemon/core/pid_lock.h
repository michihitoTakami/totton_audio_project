#pragma once

#include <optional>
#include <string>

namespace daemon_core {

class PidLock {
   public:
    static std::optional<PidLock> tryAcquire(const std::string& path);

    PidLock(const PidLock&) = delete;
    PidLock& operator=(const PidLock&) = delete;

    PidLock(PidLock&& other) noexcept;
    PidLock& operator=(PidLock&& other) noexcept;

    ~PidLock();

    const std::string& path() const;

   private:
    PidLock(std::string path, int fd);

    void release() noexcept;

    std::string path_;
    int fd_ = -1;
};

}  // namespace daemon_core
