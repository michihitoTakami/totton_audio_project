#include "daemon/core/pid_lock.h"

#include "logging/logger.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <optional>
#include <sys/file.h>
#include <unistd.h>

namespace daemon_core {

namespace {

pid_t readPidFromLockfile(const std::string& path) {
    std::ifstream pidfile(path);
    if (!pidfile.is_open()) {
        return 0;
    }
    pid_t pid = 0;
    pidfile >> pid;
    return pid;
}

}  // namespace

std::optional<PidLock> PidLock::tryAcquire(const std::string& path) {
    int fd = open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        LOG_ERROR("Cannot open PID file: {} ({})", path, strerror(errno));
        return std::nullopt;
    }

    if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
        if (errno == EWOULDBLOCK) {
            pid_t existingPid = readPidFromLockfile(path);
            if (existingPid > 0) {
                LOG_ERROR("Another instance is already running (PID: {})", existingPid);
            } else {
                LOG_ERROR("Another instance is already running");
            }
            LOG_ERROR("Use './scripts/daemon.sh stop' to stop it.");
        } else {
            LOG_ERROR("Cannot lock PID file: {}", strerror(errno));
        }
        close(fd);
        return std::nullopt;
    }

    if (ftruncate(fd, 0) < 0) {
        LOG_WARN("Cannot truncate PID file");
    }
    dprintf(fd, "%d\n", getpid());
    fsync(fd);

    return PidLock(path, fd);
}

PidLock::PidLock(std::string path, int fd) : path_(std::move(path)), fd_(fd) {}

PidLock::PidLock(PidLock&& other) noexcept : path_(std::move(other.path_)), fd_(other.fd_) {
    other.fd_ = -1;
}

PidLock& PidLock::operator=(PidLock&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release();
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    other.fd_ = -1;
    return *this;
}

PidLock::~PidLock() {
    release();
}

const std::string& PidLock::path() const {
    return path_;
}

void PidLock::release() noexcept {
    if (fd_ < 0) {
        return;
    }
    flock(fd_, LOCK_UN);
    close(fd_);
    fd_ = -1;
    if (!path_.empty()) {
        unlink(path_.c_str());
    }
}

}  // namespace daemon_core
