#include "daemon/core/pid_lock.h"
#include "gtest/gtest.h"

#include <cctype>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

class PidLockTest : public ::testing::Test {
   protected:
    fs::path tempDir;

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = "pid_lock";
        if (info) {
            name = std::string(info->test_suite_name()) + "_" + std::string(info->name());
        }
        for (char& c : name) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
                c = '_';
            }
        }
        tempDir = fs::temp_directory_path() /
                  ("gpu_upsampler_test_" + name + "_" + std::to_string(getpid()));
        fs::create_directories(tempDir);
    }

    void TearDown() override {
        fs::remove_all(tempDir);
    }
};

TEST_F(PidLockTest, AcquireAndReleaseRemovesFile) {
    fs::path lockPath = tempDir / "gpu_upsampler.pid";

    auto lock = daemon_core::PidLock::tryAcquire(lockPath.string());
    ASSERT_TRUE(lock.has_value());
    EXPECT_TRUE(fs::exists(lockPath));

    lock.reset();
    EXPECT_FALSE(fs::exists(lockPath));
}

TEST_F(PidLockTest, SecondProcessCannotAcquireWhileLocked) {
    fs::path lockPath = tempDir / "gpu_upsampler.pid";

    auto lock = daemon_core::PidLock::tryAcquire(lockPath.string());
    ASSERT_TRUE(lock.has_value());

    pid_t pid = fork();
    ASSERT_GE(pid, 0);

    if (pid == 0) {
        auto second = daemon_core::PidLock::tryAcquire(lockPath.string());
        _exit(second.has_value() ? 1 : 0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}
